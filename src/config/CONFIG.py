# Configuration
CONFIG = {
    'data_path': "data/raw/iot23_combined.csv",
    'blocksize': '64MB',
    'k_range': range(2, 11),
    'numerical_columns': [
        'duration', 'orig_bytes', 'resp_bytes', 'missed_bytes', 
        'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes',
        'proto_icmp', 'proto_tcp', 'proto_udp',
        'conn_state_OTH', 'conn_state_REJ', 'conn_state_RSTO',
        'conn_state_RSTOS0', 'conn_state_RSTR', 'conn_state_RSTRH',
        'conn_state_S0', 'conn_state_S1', 'conn_state_S2',
        'conn_state_S3', 'conn_state_SF', 'conn_state_SH',
        'conn_state_SHR'
    ]
    # 'numerical_columns': [
    #     'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
    #     'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
    #     'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean',
    #     'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl',
    #     'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
    #     'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst',
    #     'is_sm_ips_ports', 'label'
    # ]
}