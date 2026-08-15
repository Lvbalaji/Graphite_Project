import argparse
import pathlib

def param_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--N', 
                        nargs=1, type=int, default=4,
                        help='N for Graphite N-gram')

    parser.add_argument('--pool', 
                        nargs=1, type=str, choices=["sum", "mean", "max"], default="sum",
                        help='pooling method for Graphite N-gram')

    parser.add_argument('--classifier',
                        type=str,
                        choices=["rf", "xgboost", "lightgbm", "mlp", "all"],
                        default="all",
                        help='Classifier: rf, xgboost, lightgbm, mlp, or all (default: all)')

    parser.add_argument('--dataset-path', 
                        nargs='?', type=str,
                        default=str(pathlib.Path(__file__).parent.parent.joinpath("dataset")),
                        help='processed-pickle dataset directory-path')

    parser.add_argument('--eventname-edgefeats-path', 
                        nargs='?', type=str,
                        default=str(pathlib.Path(__file__).parent.parent.joinpath("dataset/EventName_EdgeFeatures.json")),
                        help='event-name edge-features as json')

    parser.add_argument('--nodetype-nodefeats-path', 
                        nargs='?', type=str,
                        default=str(pathlib.Path(__file__).parent.parent.joinpath("dataset/NodeType_NodeFeatures.json")),
                        help='node-type node-features as json')

    parser.add_argument('--output-dir',
                        type=str, default=str(pathlib.Path(__file__).parent.joinpath("results")),
                        help='directory to save comparative analysis results')

    return parser.parse_args()
