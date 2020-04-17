
import paddle
import paddle.fluid as fluid

# 定义cnn模型
# 其中class_dim表示分类的类别数，win_sizes表示使用卷积核窗口大小
def cnn_net(data,
            label,
            dict_dim,
            emb_dim=128,
            hid_dim=128,
            hid_dim2=96,
            class_dim=3,
            win_size=3,
            is_prediction=False):
    """
    Conv net
    """
    # embedding layer
    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])

    # convolution layer
    conv_3 = fluid.nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=win_size,
        act="tanh",
        pool_type="max")

    # full connect layer
    fc_1 = fluid.layers.fc(input=[conv_3], size=hid_dim2)
    # softmax layer
    prediction = fluid.layers.fc(input=[fc_1], size=class_dim, act="softmax")
    if is_prediction:
        return prediction
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)

    return avg_cost, prediction