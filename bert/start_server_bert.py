from bert_serving.server.helper import get_args_parser,get_shutdown_parser
from bert_serving.server import BertServer

def start_server(max_seq_len,pretrained_model):
    args = get_args_parser().parse_args(['-model_dir', pretrained_model,
                                         '-port', '5555',
                                         '-port_out', '5556',
                                         '-pooling_strategy', 'NONE',
                                         '-show_tokens_to_client',
                                         '-max_seq_len', str(max_seq_len),
                                         '-mask_cls_sep',
                                         '-cpu'])
    server = BertServer(args)
    server.start()

def stop_server():
    shut_args = get_shutdown_parser().parse_args(['-port','5555'])
    BertServer.shutdown(shut_args)
