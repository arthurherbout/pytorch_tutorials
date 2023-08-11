from tutorials.fashion_mnist import model, dataset
from tutorials import utils
from torch import nn, optim


if __name__ == "__main__":
    # constants
    batch_size = 64
    learning_rate = 1e-3
    epochs = 5
    loss_fn = nn.CrossEntropyLoss()


    device = utils.get_device_for_training()

    n_network = model.FashionMnistNetwork().to(device)

    optimizer = optim.SGD(n_network.parameters(), lr=learning_rate)

    train_dataloader = dataset.load_training_data(batch_size=batch_size)
    test_dataloader = dataset.load_testing_data(batch_size=batch_size)

    # learning
    training_losses = []
    testing_losses = []
    testing_accuracies = []

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        training_loss = utils.train_loop(dataloader=train_dataloader,
                                         model=n_network,
                                         loss_fn=loss_fn,
                                         optimizer=optimizer,
                                         device=device)
        testing_metrics = utils.test_loop(dataloader=test_dataloader,
                                          model=n_network,
                                          loss_fn=loss_fn,
                                          device=device)

        # store results
        training_losses.append(training_loss)
        testing_losses.append(testing_metrics["loss"])
        testing_accuracies.append(testing_metrics["accuracy"])

    print("This is the training loss:")
    print(training_losses)

    print("This is the testing loss:")
    print(testing_losses)

    print("This is the testing accuracy:")
    print(testing_accuracies)

