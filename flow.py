#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function
from bcc import BPF
from time import sleep
import argparse
from collections import namedtuple, defaultdict
from kafka import KafkaProducer
from kafka.errors import KafkaError
import json
import time
import random

SessionKey = namedtuple('Session', ['pid'])
producer = KafkaProducer(bootstrap_servers=['192.168.31.106:9092'],
                         value_serializer=lambda m: json.dumps(m).encode('ascii'))


def get_ipv4_session_key(k):
    return SessionKey(pid=k.pid)


def range_check(string):
    value = int(string)
    if value < 1:
        msg = "value must be stricly positive, got %d" % (value,)
        raise argparse.ArgumentTypeError(msg)
    return value


def pid_to_command(pid):
    try:
        cgroup_info = open("/proc/%s/cgroup" % pid, "r").read().rstrip()
        # if 'ac4f856011bb3f0a5992b43432c5bc16a0f6e15825fea4799259f3d3d0e6666d' in cgroup_info: # use `python flow.py` can see this print
        # print("=============="+str(pid)+"=================")
        command = open("/proc/%s/comm" % pid, "r").read().rstrip()
        return command
    except IOError:
        return str(pid)


def sink_flow(pid, recv_bytes, send_bytes, sum_recv, sum_send, sum_kb):
    print("%-10d %-12.12s %-10d %-10d %-10d %-10d %-10d" % (
        k.pid, pid_to_command(k.pid), recv_bytes, send_bytes, sum_recv, sum_send, sum_kb))
    current_time = int(time.time())
    message = {"time": current_time + i, "speed": send_bytes}
    print('sending {}'.format(message))
    producer.send('test1', message)


if __name__ == '__main__':

    examples = """examples:
        python flow.py          # trace send/recv flow by host 
        python flow.py -p 15533   # only trace PID 15533
    """

    parser = argparse.ArgumentParser(
        description="Summarize send and recv flow by host",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples
    )
    parser.add_argument("-p", "--pid",
                        help="Trace this pid only")
    parser.add_argument("interval", nargs="?", default=1, type=range_check,
                        help="output interval, in second (default 1)")
    parser.add_argument("count", nargs="?", default=-1, type=range_check,
                        help="number of outputs")
    args = parser.parse_args()
    print("args.interval is: %x s" % args.interval)

    bpf_program = """
    #ifdef asm_inline
    #undef asm_inline
    #define asm_inline asm
    #endif

    #include <uapi/linux/ptrace.h>
    #include <net/sock.h>
    #include <bcc/proto.h>

    struct ipv4_key_t {
        u32 pid;
    };

    BPF_HASH(ipv4_send_bytes, struct ipv4_key_t);
    BPF_HASH(ipv4_recv_bytes, struct ipv4_key_t);

    int kprobe__tcp_sendmsg(struct pt_regs *ctx, struct sock *sk,
        struct msghdr *msg, size_t size)
    {
        u32 pid = bpf_get_current_pid_tgid() >> 32;
        FILTER_PID
        u16 family = sk->__sk_common.skc_family;
        if (family == AF_INET) {
            struct ipv4_key_t ipv4_key = {.pid = pid};
            ipv4_send_bytes.increment(ipv4_key, size);
        }
        return 0;
    }

    int kprobe__tcp_cleanup_rbuf(struct pt_regs *ctx, struct sock *sk, int copied)
    {
        u32 pid = bpf_get_current_pid_tgid() >> 32;
        FILTER_PID
        u16 family = sk->__sk_common.skc_family;
        u64 *val, zero =0;
        if (copied <= 0)
            return 0;
        if (family == AF_INET) {
            struct ipv4_key_t ipv4_key = {.pid = pid};
            ipv4_recv_bytes.increment(ipv4_key, copied);
        }
        return 0;
    }
    """

    if args.pid:
        bpf_program = bpf_program.replace('FILTER_PID',
                                          'if (pid != %s) { return 0; }' % args.pid)
    else:
        bpf_program = bpf_program.replace('FILTER_PID', '')

    # init bpf
    b = BPF(text=bpf_program)
    ipv4_send_bytes = b["ipv4_send_bytes"]
    # print(type(ipv4_send_bytes)) <class 'bcc.table.HashTable'>
    ipv4_recv_bytes = b["ipv4_recv_bytes"]

    # header
    print("%-10s %-12s %-10s %-10s %-10s %-10s %-10s" % (
    "PID", "COMM", "RX_KB", "TX_KB", "RXSUM_KB", "TXSUM_KB", "SUM_KB"))

    # output
    sum_recv = 0
    sum_send = 0
    sum_kb = 0
    i = 0
    interrupted = False

    while i != args.count and not interrupted:
        try:
            sleep(args.interval)
        except KeyboardInterrupt:
            interrupted = True

        ipv4_throughput = defaultdict(lambda: [0, 0])
        for k, v in ipv4_send_bytes.items():
            key = get_ipv4_session_key(k)
            ipv4_throughput[key][0] = v.value
        ipv4_send_bytes.clear()

        for k, v in ipv4_recv_bytes.items():
            key = get_ipv4_session_key(k)
            ipv4_throughput[key][1] = v.value
        ipv4_recv_bytes.clear()

        if ipv4_throughput:
            for k, (send_bytes, recv_bytes) in sorted(ipv4_throughput.items(),
                                                      key=lambda kv: sum(kv[1]),
                                                      reverse=True):
                recv_bytes = int(recv_bytes)  # /1024
                send_bytes = int(send_bytes)  # /1024
                sum_recv += recv_bytes
                sum_send += send_bytes
                sum_kb = sum_recv + sum_send
                sink_flow(k.pid, recv_bytes, send_bytes, sum_recv, sum_send, sum_kb)
        else:
            print("%-10d %-12.12s %-10d %-10d %-10d %-10d %-10d" % (0, "-", 0, 0, sum_recv, sum_send, sum_kb))
        i += 1
