#include "rdma_service.h"
#include "include/gpu_mem_util.h"
#include <chrono>

RdmaService::RdmaService(char* server_name, int memory_size) {
    memset(&config, 0, sizeof(config));
    config.tcp_port = 18515;
    config.ib_port = 1;
    config.gid_idx = -1;
    config.server_name = server_name;
    this->memory_size = memory_size;
    memset(&res, 0, sizeof(res));
    res.sock = -1;
    if (server_name) {
        res.buf = (char*)work_buffer_alloc(memory_size, 1, "5b:00.0");
    } else {
        res.buf = (char*)work_buffer_alloc(memory_size, 0, nullptr);
    }
}

RdmaService::~RdmaService() {
    if (res.qp) {
        ibv_destroy_qp(res.qp);
        res.qp = NULL;
    }
    if (res.mr) {
        ibv_dereg_mr(res.mr);
        res.mr = NULL;
    }
    if (res.buf) {
        if (config.server_name) {
            work_buffer_free(res.buf, 1);
        } else {
            work_buffer_free(res.buf, 0);
        }
        res.buf = NULL;
    }
    if (res.cq) {
        ibv_destroy_cq(res.cq);
        res.cq = NULL;
    }
    if (res.pd) {
        ibv_dealloc_pd(res.pd);
        res.pd = NULL;
    }
    if (res.ib_ctx) {
        ibv_close_device(res.ib_ctx);
        res.ib_ctx = NULL;
    }
    if (res.sock >= 0) {
        if (close(res.sock))
            fprintf(stderr, "failed to close socket\n");
        res.sock = -1;
    }
}

int RdmaService::resources_create() {
    struct resources *res = &this->res;
    struct ibv_device **dev_list = NULL;
    struct ibv_qp_init_attr qp_init_attr;
    struct ibv_device *ib_dev = NULL;
    size_t size;
    int i;
    int mr_flags = 0;
    int cq_size = 0;
    int num_devices;
    int rc = 0;
    /* if client side */
    if (config.server_name) {
        res->sock = sock_connect(config.server_name, config.tcp_port);
        if (res->sock < 0) {
            fprintf(
                stderr,
                "failed to establish TCP connection to server %s, port %d\n",
                config.server_name, config.tcp_port);
            rc = -1;
            goto resources_create_exit;
        }
    } else {
        fprintf(stdout, "waiting on port %d for TCP connection\n",
                config.tcp_port);
        res->sock = sock_connect(NULL, config.tcp_port);
        if (res->sock < 0) {
            fprintf(
                stderr,
                "failed to establish TCP connection with client on port %d\n",
                config.tcp_port);
            rc = -1;
            goto resources_create_exit;
        }
    }
    fprintf(stdout, "TCP connection was established\n");
    fprintf(stdout, "searching for IB devices in host\n");
    /* get device names in the system */
    dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list) {
        fprintf(stderr, "failed to get IB devices list\n");
        rc = 1;
        goto resources_create_exit;
    }
    /* if there isn't any IB device in host */
    if (!num_devices) {
        fprintf(stderr, "found %d device(s)\n", num_devices);
        rc = 1;
        goto resources_create_exit;
    }
    fprintf(stdout, "found %d device(s)\n", num_devices);
    /* search for the specific device we want to work with */
    for (i = 0; i < num_devices; i++) {
        if (!config.dev_name) {
            config.dev_name = strdup(ibv_get_device_name(dev_list[i]));
            fprintf(stdout, "device not specified, using first one found: %s\n",
                    config.dev_name);
        }
        if (!strcmp(ibv_get_device_name(dev_list[i]), config.dev_name)) {
            ib_dev = dev_list[i];
            break;
        }
    }
    /* if the device wasn't found in host */
    if (!ib_dev) {
        fprintf(stderr, "IB device %s wasn't found\n", config.dev_name);
        rc = 1;
        goto resources_create_exit;
    }
    /* get device handle */
    res->ib_ctx = ibv_open_device(ib_dev);
    if (!res->ib_ctx) {
        fprintf(stderr, "failed to open device %s\n", config.dev_name);
        rc = 1;
        goto resources_create_exit;
    }
    /* We are now done with device list, free it */
    ibv_free_device_list(dev_list);
    dev_list = NULL;
    ib_dev = NULL;
    /* query port properties */
    if (ibv_query_port(res->ib_ctx, config.ib_port, &res->port_attr)) {
        fprintf(stderr, "ibv_query_port on port %u failed\n", config.ib_port);
        rc = 1;
        goto resources_create_exit;
    }
    /* allocate Protection Domain */
    res->pd = ibv_alloc_pd(res->ib_ctx);
    if (!res->pd) {
        fprintf(stderr, "ibv_alloc_pd failed\n");
        rc = 1;
        goto resources_create_exit;
    }
    /* each side will send only one WR, so Completion Queue with 1 entry is
     * enough */
    cq_size = 1;
    res->cq = ibv_create_cq(res->ib_ctx, cq_size, NULL, NULL, 0);
    if (!res->cq) {
        fprintf(stderr, "failed to create CQ with %u entries\n", cq_size);
        rc = 1;
        goto resources_create_exit;
    }
    /* allocate the memory buffer that will hold the data */
    size = memory_size;
    mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
               IBV_ACCESS_REMOTE_WRITE;
    res->mr = ibv_reg_mr(res->pd, res->buf, size, mr_flags);
    if (!res->mr) {
        fprintf(stderr, "ibv_reg_mr failed with mr_flags=0x%x\n", mr_flags);
        rc = 1;
        goto resources_create_exit;
    }
    fprintf(
        stdout,
        "MR was registered with addr=%p, lkey=0x%x, rkey=0x%x, flags=0x%x\n",
        res->buf, res->mr->lkey, res->mr->rkey, mr_flags);
    /* create the Queue Pair */
    memset(&qp_init_attr, 0, sizeof(qp_init_attr));
    qp_init_attr.qp_type = IBV_QPT_RC;
    qp_init_attr.sq_sig_all = 1;
    qp_init_attr.send_cq = res->cq;
    qp_init_attr.recv_cq = res->cq;
    qp_init_attr.cap.max_send_wr = 1;
    qp_init_attr.cap.max_recv_wr = 1;
    qp_init_attr.cap.max_send_sge = 1;
    qp_init_attr.cap.max_recv_sge = 1;
    res->qp = ibv_create_qp(res->pd, &qp_init_attr);
    if (!res->qp) {
        fprintf(stderr, "failed to create QP\n");
        rc = 1;
        goto resources_create_exit;
    }
    fprintf(stdout, "QP was created, QP number=0x%x\n", res->qp->qp_num);
resources_create_exit:
    if (rc) {
        /* Error encountered, cleanup */
        if (res->qp) {
            ibv_destroy_qp(res->qp);
            res->qp = NULL;
        }
        if (res->mr) {
            ibv_dereg_mr(res->mr);
            res->mr = NULL;
        }
        if (res->buf) {
            if (config.server_name) {
                work_buffer_free(res->buf, 1);
            } else {
                work_buffer_free(res->buf, 0);
            }
            res->buf = NULL;
        }
        if (res->cq) {
            ibv_destroy_cq(res->cq);
            res->cq = NULL;
        }
        if (res->pd) {
            ibv_dealloc_pd(res->pd);
            res->pd = NULL;
        }
        if (res->ib_ctx) {
            ibv_close_device(res->ib_ctx);
            res->ib_ctx = NULL;
        }
        if (dev_list) {
            ibv_free_device_list(dev_list);
            dev_list = NULL;
        }
        if (res->sock >= 0) {
            if (close(res->sock))
                fprintf(stderr, "failed to close socket\n");
            res->sock = -1;
        }
    }
    return rc;
}

int RdmaService::connect_qp() {
    struct resources *res = &this->res;
    struct cm_con_data_t local_con_data;
    struct cm_con_data_t remote_con_data;
    struct cm_con_data_t tmp_con_data;
    int rc = 0;
    char temp_char;
    union ibv_gid my_gid;
    if (config.gid_idx >= 0) {
        rc =
            ibv_query_gid(res->ib_ctx, config.ib_port, config.gid_idx, &my_gid);
        if (rc) {
            fprintf(stderr, "could not get gid for port %d, index %d\n",
                    config.ib_port, config.gid_idx);
            return rc;
        }
    } else
        memset(&my_gid, 0, sizeof my_gid);
    /* exchange using TCP sockets info required to connect QPs */
    local_con_data.addr = htonll((uintptr_t)res->buf);
    local_con_data.rkey = htonl(res->mr->rkey);
    local_con_data.qp_num = htonl(res->qp->qp_num);
    local_con_data.lid = htons(res->port_attr.lid);
    memcpy(local_con_data.gid, &my_gid, 16);
    fprintf(stdout, "\nLocal LID = 0x%x\n", res->port_attr.lid);
    if (sock_sync_data(res->sock, sizeof(struct cm_con_data_t),
                       (char *)&local_con_data, (char *)&tmp_con_data) < 0) {
        fprintf(stderr, "failed to exchange connection data between sides\n");
        rc = 1;
        goto connect_qp_exit;
    }
    remote_con_data.addr = ntohll(tmp_con_data.addr);
    remote_con_data.rkey = ntohl(tmp_con_data.rkey);
    remote_con_data.qp_num = ntohl(tmp_con_data.qp_num);
    remote_con_data.lid = ntohs(tmp_con_data.lid);
    memcpy(remote_con_data.gid, tmp_con_data.gid, 16);
    /* save the remote side attributes, we will need it for the post SR */
    res->remote_props = remote_con_data;
    fprintf(stdout, "Remote address = 0x%" PRIx64 "\n", remote_con_data.addr);
    fprintf(stdout, "Remote rkey = 0x%x\n", remote_con_data.rkey);
    fprintf(stdout, "Remote QP number = 0x%x\n", remote_con_data.qp_num);
    fprintf(stdout, "Remote LID = 0x%x\n", remote_con_data.lid);
    if (config.gid_idx >= 0) {
        uint8_t *p = remote_con_data.gid;
        fprintf(stdout,
                "Remote GID "
                "=%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%"
                "02x:%02x:%02x:%02x\n ",
                p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
                p[10], p[11], p[12], p[13], p[14], p[15]);
    }
    /* modify the QP to init */
    rc = modify_qp_to_init(res->qp, config.ib_port);
    if (rc) {
        fprintf(stderr, "change QP state to INIT failed\n");
        goto connect_qp_exit;
    }
    /* let the client post RR to be prepared for incoming messages */
    if (config.server_name) {
        rc = post_receive();
        if (rc) {
            fprintf(stderr, "failed to post RR\n");
            goto connect_qp_exit;
        }
    }
    /* modify the QP to RTR */
    rc = modify_qp_to_rtr(res->qp, remote_con_data.qp_num, remote_con_data.lid,
                          remote_con_data.gid, config.ib_port, config.gid_idx);
    if (rc) {
        fprintf(stderr, "failed to modify QP state to RTR\n");
        goto connect_qp_exit;
    }
    rc = modify_qp_to_rts(res->qp);
    if (rc) {
        fprintf(stderr, "failed to modify QP state to RTR\n");
        goto connect_qp_exit;
    }
    fprintf(stdout, "QP state was change to RTS\n");
    /* sync to make sure that both sides are in states that they can connect to
     * prevent packet loose */
    if (sock_sync_data(res->sock, 1, "Q",
                       &temp_char)) /* just send a dummy char back and forth */
    {
        fprintf(stderr, "sync error after QPs are were moved to RTS\n");
        rc = 1;
    }
connect_qp_exit:
    return rc;
}

int RdmaService::post_receive() {
    struct resources *res = &this->res;
    struct ibv_recv_wr rr;
    struct ibv_sge sge;
    struct ibv_recv_wr *bad_wr;
    int rc;
    /* prepare the scatter/gather entry */
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uintptr_t)res->buf;
    sge.length = memory_size;
    sge.lkey = res->mr->lkey;
    /* prepare the receive work request */
    memset(&rr, 0, sizeof(rr));
    rr.next = NULL;
    rr.wr_id = 0;
    rr.sg_list = &sge;
    rr.num_sge = 1;
    /* post the Receive Request to the RQ */
    rc = ibv_post_recv(res->qp, &rr, &bad_wr);
    if (rc)
        fprintf(stderr, "failed to post RR\n");
    else
        fprintf(stdout, "Receive Request was posted\n");
    return rc;
}

int RdmaService::post_send(ibv_wr_opcode opcode) {
    struct resources *res = &this->res;
    struct ibv_send_wr sr;
    struct ibv_sge sge;
    struct ibv_send_wr *bad_wr = NULL;
    int rc;
    /* prepare the scatter/gather entry */
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uintptr_t)res->buf;
    sge.length = memory_size;
    sge.lkey = res->mr->lkey;
    /* prepare the send work request */
    memset(&sr, 0, sizeof(sr));
    sr.next = NULL;
    sr.wr_id = 0;
    sr.sg_list = &sge;
    sr.num_sge = 1;
    sr.opcode = opcode;
    sr.send_flags = IBV_SEND_SIGNALED;
    if (opcode != IBV_WR_SEND) {
        sr.wr.rdma.remote_addr = res->remote_props.addr;
        sr.wr.rdma.rkey = res->remote_props.rkey;
    }
    /* there is a Receive Request in the responder side, so we won't get any
     * into RNR flow */
    rc = ibv_post_send(res->qp, &sr, &bad_wr);
    if (rc)
        fprintf(stderr, "failed to post SR\n");
    else {
        switch (opcode) {
        case IBV_WR_SEND:
            fprintf(stdout, "Send Request was posted\n");
            break;
        case IBV_WR_RDMA_READ:
            fprintf(stdout, "RDMA Read Request was posted\n");
            break;
        case IBV_WR_RDMA_WRITE:
            fprintf(stdout, "RDMA Write Request was posted\n");
            break;
        default:
            fprintf(stdout, "Unknown Request was posted\n");
            break;
        }
    }
    return rc;
}

int RdmaService::poll_completion() {
    struct resources *res = &this->res;
    struct ibv_wc wc;
    unsigned long start_time_msec;
    unsigned long cur_time_msec;
    struct timeval cur_time;
    int poll_result;
    int rc = 0;
    /* poll the completion for a while before giving up of doing it .. */
    gettimeofday(&cur_time, NULL);
    start_time_msec = (cur_time.tv_sec * 1000) + (cur_time.tv_usec / 1000);
    do {
        poll_result = ibv_poll_cq(res->cq, 1, &wc);
        gettimeofday(&cur_time, NULL);
        cur_time_msec = (cur_time.tv_sec * 1000) + (cur_time.tv_usec / 1000);
    } while ((poll_result == 0) &&
             ((cur_time_msec - start_time_msec) < MAX_POLL_CQ_TIMEOUT));
    if (poll_result < 0) {
        /* poll CQ failed */
        fprintf(stderr, "poll CQ failed\n");
        rc = 1;
    } else if (poll_result == 0) { /* the CQ is empty */
        fprintf(stderr, "completion wasn't found in the CQ after timeout\n");
        rc = 1;
    } else {
        /* CQE found */
        fprintf(stdout, "completion was found in CQ with status 0x%x\n",
                wc.status);
        /* check the completion status (here we don't care about the completion
         * opcode */
        if (wc.status != IBV_WC_SUCCESS) {
            fprintf(
                stderr,
                "got bad completion with status: 0x%x, vendor syndrome: 0x%x\n",
                wc.status, wc.vendor_err);
            rc = 1;
        }
    }
    return rc;
}