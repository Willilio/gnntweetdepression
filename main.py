# Imports
import os
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.gen_datasets import *
from src.model import *
# from evaluate_results import evaluate_model_results
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)


def load_datasets(args):
    """Loads dataset and graph if exists, else create and process them from raw data
    Returns --->
    f: torch tensor input of GCN (Identity matrix)
    X: input of GCN (Identity matrix)
    A_hat: transformed adjacency matrix A
    selected: indexes of selected labelled nodes for training
    test_idxs: indexes of not-selected nodes for inference/testing
    labels_selected: labels of selected labelled nodes for training
    labels_not_selected: labels of not-selected labelled nodes for inference/testing
    """

    # Loading graph data
    logger.info("Loading data...")
    df_data_path = DATAFOLDER + "gen/df_data.pkl"
    graph_path = DATAFOLDER + "gen/text_graph.pkl"
    if not os.path.isfile(df_data_path) or not os.path.isfile(graph_path):  # when graph isn't yet built
        logger.info("Building datasets and graph from raw data... Note this will take quite a while...")
        generate_text_graph()
    df_data = load_pickle("gen/df_data.pkl")
    graph = load_pickle("gen/text_graph.pkl")

    # Adjacency matrix and degree matrices
    logger.info("Building adjacency and degree matrices...")
    adjacency = nx.to_numpy_array(graph, weight="weight")
    adjacency = adjacency + np.eye(graph.number_of_nodes())
    degrees = []
    for d in graph.degree(weight=None):
        if d == 0:
            degrees.append(0)
        else:
            degrees.append(d[1] ** (-0.5))
    degrees = np.diag(degrees)
    x = np.eye(graph.number_of_nodes())  # Features are just identity matrix
    a_hat = degrees @ adjacency @ degrees
    f_ = x  # (n X n) X (n X n) x (n X n) X (n X n) input of net

    # Split testing and training data
    logger.info("Splitting labels for training and inferring...")
    test_indices = []
    for target_id in df_data["target"].unique():
        dum = df_data[df_data["target"] == target_id]
        if len(dum) >= 4:
            test_indices.extend(
                list(np.random.choice(dum.index, size=round(args.test_ratio * len(dum)), replace=False)))
    save_as_pickle("gen/test_idxs.pkl", test_indices)

    # select only certain labelled nodes for semi-supervised GCN
    sel = []
    for i in range(len(df_data)):
        if i not in test_indices:
            sel.append(i)
    save_as_pickle("gen/selected.pkl", sel)

    # Categorize selected labels
    labels_sel = [l for idx, l in enumerate(df_data["target"]) if idx in sel]
    labels_not_sel = [l for idx, l in enumerate(df_data["target"]) if idx not in sel]
    f_ = torch.from_numpy(f_).float()
    save_as_pickle("gen/labels_selected.pkl", labels_sel)
    save_as_pickle("gen/labels_not_selected.pkl", labels_not_sel)
    logger.info("Split into %d train and %d test labels." % (len(labels_sel), len(labels_not_sel)))
    return f_, x, a_hat, sel, labels_sel, labels_not_sel, test_indices


def load_state(n_net, opt, sch, model_no=0, load_best=False):
    """ Loads saved model and optimizer states if exists """
    logger.info("Initializing model and optimizer states...")
    checkpoint_path = os.path.join(DATAFOLDER, "gen/test_checkpoint_%d.pth.tar" % model_no)
    best_path = os.path.join(DATAFOLDER, "gen/test_model_best_%d.pth.tar" % model_no)
    epoch_start, pred_best, checkpoint = 0, 0, None
    if load_best and os.path.isfile(best_path):
        checkpoint = torch.load(best_path)
        logger.info("Loaded best model.")
    elif os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        logger.info("Loaded checkpoint model.")
    if checkpoint is not None:
        epoch_start = checkpoint['epoch']
        pred_best = checkpoint['best_acc']
        n_net.load_state_dict(checkpoint['state_dict'])
        opt.load_state_dict(checkpoint['optimizer'])
        sch.load_state_dict(checkpoint['scheduler'])
        logger.info("Loaded model and optimizer.")
    return epoch_start, pred_best


def load_results(model_no=0):
    """ Loads saved results if exists """
    losses_path = "./data/gen/test_losses_per_epoch_%d.pkl" % model_no
    accuracy_path = "./data/gen/test_accuracy_per_epoch_%d.pkl" % model_no
    if os.path.isfile(losses_path) and os.path.isfile(accuracy_path):
        losses_each_epoch = load_pickle("gen/test_losses_per_epoch_%d.pkl" % model_no)
        accuracy_per_epoch = load_pickle("gen/test_accuracy_per_epoch_%d.pkl" % model_no)
        logger.info("Loaded results buffer")
    else:
        losses_each_epoch, accuracy_per_epoch = [], []
    return losses_each_epoch, accuracy_per_epoch


def evaluate(out, labels_e):
    _, labels = out.max(1)
    labels = labels.numpy()
    return sum([(le - 1) for le in labels_e] == labels) / len(labels)


if __name__ == "__main__":
    gcn_args = GCNArgs()
    gcn_args.hidden_sizes = [1024, 2048, 1024]
    gcn_args.num_classes = 2
    gcn_args.test_ratio = 0.1
    gcn_args.num_epochs = 4000
    gcn_args.learning_rate = 0.11
    save_as_pickle("gen/args.pkl", gcn_args)

    f, X, A_hat, selected, labels_selected, labels_not_selected, test_idxs = load_datasets(gcn_args)
    net = GCN(X.shape[1], A_hat, gcn_args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=gcn_args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 2000, 3000, 4000, 5000, 6000], gamma=0.77)

    start_epoch, best_pred = load_state(net, optimizer, scheduler, model_no=gcn_args.model_id, load_best=True)
    losses_per_epoch, evaluation_untrained = load_results(model_no=gcn_args.model_id)

    logger.info("Starting training process...")
    net.train()
    evaluation_trained = []
    for e in range(start_epoch, gcn_args.num_epochs):
        optimizer.zero_grad()
        output = net(f)  # TODO: Always returning the same value regardless of input
        loss = criterion(output[selected], torch.tensor(labels_selected).long())
        losses_per_epoch.append(loss.item())
        loss.backward()
        optimizer.step()
        # Evaluate the nn
        net.eval()
        with torch.no_grad():
            pred_labels = net(f)
            trained_accuracy = evaluate(output[selected], labels_selected)
            untrained_accuracy = evaluate(pred_labels[test_idxs], labels_not_selected)
        evaluation_trained.append((e, trained_accuracy))
        evaluation_untrained.append((e, untrained_accuracy))
        print("[Epoch %d]: Evaluation accuracy of trained nodes: %.7f" % (e, trained_accuracy))
        print("[Epoch %d]: Evaluation accuracy of test nodes: %.7f" % (e, untrained_accuracy))
        print("Labels of trained nodes: \n", output[selected].max(1)[1])
        net.train()
        if trained_accuracy > best_pred:
            best_pred = trained_accuracy
            torch.save({
                'epoch': e + 1,
                'state_dict': net.state_dict(),
                'best_acc': trained_accuracy,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, os.path.join(DATAFOLDER, "gen/test_model_best_%d.pth.tar" % gcn_args.model_id))
        if (e % 25) == 0:
            save_as_pickle("gen/test_losses_per_epoch_%d.pkl" % gcn_args.model_id, losses_per_epoch)
            save_as_pickle("gen/test_accuracy_per_epoch_%d.pkl" % gcn_args.model_id, evaluation_untrained)
            torch.save({
                'epoch': e + 1,
                'state_dict': net.state_dict(),
                'best_acc': trained_accuracy,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, os.path.join(DATAFOLDER, "gen/est_checkpoint_%d.pth.tar" % gcn_args.model_id))
        scheduler.step()

    logger.info("Finished training!")
    evaluation_trained = np.array(evaluation_trained)
    evaluation_untrained = np.array(evaluation_untrained)
    save_as_pickle("gen/test_losses_per_epoch_%d_final.pkl" % gcn_args.model_id, losses_per_epoch)
    save_as_pickle("gen/train_accuracy_per_epoch_%d_final.pkl" % gcn_args.model_id, evaluation_trained)
    save_as_pickle("gen/test_accuracy_per_epoch_%d_final.pkl" % gcn_args.model_id, evaluation_untrained)

    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(111)
    ax.scatter([i for i in range(len(losses_per_epoch))], losses_per_epoch)
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Loss", fontsize=15)
    ax.set_title("Loss vs Epoch", fontsize=20)
    plt.savefig(os.path.join(DATAFOLDER, "gen/loss_vs_epoch.png"))

    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(111)
    ax.scatter(evaluation_trained[:, 0], evaluation_trained[:, 1])
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Accuracy on trained nodes", fontsize=15)
    ax.set_title("Accuracy (trained nodes) vs Epoch", fontsize=20)
    plt.savefig(os.path.join(DATAFOLDER, "gen/trained_accuracy_vs_epoch.png"))

    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(111)
    ax.scatter(evaluation_untrained[:, 0], evaluation_untrained[:, 1])
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Accuracy on untrained nodes", fontsize=15)
    ax.set_title("Accuracy (untrained nodes) vs Epoch", fontsize=20)
    plt.savefig(os.path.join(DATAFOLDER, "gen/untrained_accuracy_vs_epoch.png"))

    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(111)
    ax.scatter(evaluation_trained[:, 0], evaluation_trained[:, 1], c="red", marker="v",
               label="Trained Nodes")
    ax.scatter(evaluation_untrained[:, 0], evaluation_untrained[:, 1], c="blue", marker="o",
               label="Untrained Nodes")
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Accuracy", fontsize=15)
    ax.set_title("Accuracy vs Epoch", fontsize=20)
    ax.legend(fontsize=20)
    plt.savefig(os.path.join(DATAFOLDER, "gen/combined_plot_accuracy_vs_epoch.png"))

    # logger.info("Evaluate results...")
    # evaluate_model_results(args=gcn_args)
