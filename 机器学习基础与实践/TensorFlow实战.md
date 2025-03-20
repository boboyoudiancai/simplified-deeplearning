# TensorFlow实战

&emsp;&emsp;TensorFlow是由Google于2015年推出的一个通用的数值计算框架，它采用数据流图(Data Flow Graphs)来表示运算过程。图中的节点表示数学运算，边表示节点之间相互依赖和联系的张量（Tensor）。它可以运行在包括CPU平台，GPU平台，移动设备等在内的多种异构平台上，为深度学习的研究带来了巨大的便利。下图即为使用TensorFlow的可视化工具TensorBoard观察的一个数据流图示例。

![data flow graph exapmle](img/tensorflow-graph-example.png)

> TensorFlow支持各个主流平台上的CPU和GPU作为运算单元，关于TensorFlow的安装方法这里不在赘述，可以参考官网的[安装教程](https://www.tensorflow.org/install/)。<br>
> 同时由于目前TensorFlow还处于快速发展的时期，新版本的推出和API的改变也可能会导致示例代码失效，遇到这种情况建议根据错误提示调试或者查询官方[release说明](https://github.com/tensorflow/tensorflow/releases)。

## 1. Tensor + Flow

&emsp;&emsp;TensorFlow的名字组成很直接的揭示了其最重要的两个概念：Tensor对应的`数据`和Flow对应的`数据流图`或`计算图`。使用Tensor和Flow这两个概念，可以把机器学习算法（一个复杂的数学运算或流程）分解为通过有向图连接的简单数学操作。一般TensorFlow程序的流程也主要分为两个步骤：1）定义算法的计算图结构；2）使用计算图执行计算。

## 2. 定义计算图

### 声明Tensor

&emsp;&emsp;Tensor是TensorFLow总的基本数据类型，它与numpy中的张量（多维数组）在概念和使用上十分的类似，可以通过以下的方式声明一些张量

```python
import tensorflow as tf
import numpy as np

# 声明一个常量的张量值
x_constant = tf.constant(np.arange(24).reshape((2,3,4)), name="x_constant")
print(x_constant)
## out: Tensor("x_constant:0", shape=(2, 3, 4), dtype=int64)

# 声明一个使用shape=(3,4,5,6)，并使用高斯分布随机生成的张量
x_normal = tf.random.normal(shape=(3,4,5,6),mean=1,stddev=2,name="x_normal")
print(x_normal)
## out: Tensor("x_normal:0", shape=(3, 4, 5, 6), dtype=float32)
```

### 定义图

&emsp;&emsp;在有了tensor之后，就可以根据定义一些tensor之间的运算操作形成计算图。一个TensorFlow程序中可以有多个的计算图，TensorFlow自身会维护一个默认的计算图，因此在大部分只需要一个计算图的情况下可以不显示的对计算图进行指定，下面是一个构建简单计算图的示例：

```python
x_1 = tf.random.normal((2,3,4), name = "x_1")
x_2 = tf.random.normal((2,3,4), name = "x_2")
y_1 = x_1 + x_2
y_2 = x_1 * x_2
```

在这里，我们首先定义了两个形状为(2,3,4)的张量x_1和x_2，然后定义了y_1和y_2，他们分别由x_1和x_2相加和相乘得到，这些操作生成的计算图可以保存后使用TensorBoard展示，从图中可以非常直观的看到数据的流向过程。<br>
![simple graph example](img/simple_graph_example.png)

事实上，x_1和x_2是也是由random_normal操作生成的，将x_1展开，可以观察到它的生成过程如下所示，它是由标准正态分布根据指定的方差和均值生成。<br>
![x_1 expand](img/x_1_expand.png)

> [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)是TensorFlow配套的可视化工具，可以使用TensorBoard对TensorFlow训练过程中的参数进行可视化，上面例子中生成并展示计算图的代码方法如下：
>
> ```python
> logDir = "/your/path/to/logdir"
> train_writer = tf.summary.FileWriter(logDir,sess.graph)
> ```
> 在终端使用 tensorboard --logdir=/your/path/to/logdir即可打开TensorBoard服务，在浏览器中进行可视化。

## 3. 使用Session执行计算

&emsp;&emsp;上面所示的例子中我们已经定义了一个计算图，但是并没有真正的去执行和计算。计算图的执行需要定义TensorFlow中的一个Session，Session管理着计算所需的各种资源，通过调用Session的run方法，我们可以运行一个计算图中的操作，示例如下。

```python
tf.compat.v1.disable_eager_execution()  # 关闭即时执行模式
sess = tf.compat.v1.Session()
y_1_value, y_2_value = sess.run([y_1,y_2])

## 输出结果
# [[[ 0.99805093 -1.44938612 -0.02456892  2.45622158]
#   [-1.2138226   1.27835047  0.1263817  -1.87592912]
#   [ 1.0174129  -1.17503071  0.55202115  1.75871563]]

#  [[ 0.98850936  0.93786907  0.14832348 -3.36686707]
#   [ 0.90941292  0.28770363 -2.64593315  0.66912466]
#   [ 2.71588826 -1.45226276 -0.89671922 -1.32986116]]]
#
# [[[-0.10158142  0.35184851 -0.91852593  1.34032488]
#   [-4.20674849  0.28761467 -0.91424406  0.75015843]
#   [ 0.22345214 -0.16158691  0.04190182  0.60569668]]

#  [[ 0.06632439 -3.2975316  -0.12469446  2.80637121]
#   [ 0.20592396 -0.04410345  1.7464304  -0.05927948]
#   [ 1.24478841  0.46934795  0.19538581 -0.0220134 ]]]

```

> TensorFlow在Session启动时会默认占用全部的GPU内存资源，可以按照如下的方式指定内存占用比率
>```python
>config = tf.ConfigProto()
>config.gpu_options.per_process_gpu_memory_fraction = 0.3
>sess = tf.Session(config=config)
>```

## 4. TensorFlow中的Variable

&emsp;&emsp;机器学习算法通常需要持续的迭代更新参数变量，这些参数变量都使用TensorFlow中的`Variable`来进行保存和管理，示例如下：

```python
w_t = tf.random.normal((2,3,4),name="weight_tensor")
w = tf.Variable(w_t,name="weight")
print(w)print(w)
## out: <tf.Variable 'weight_1:0' shape=(2, 3, 4) dtype=float32_ref>
```

Variable实际上是一种特殊的运算，它接受张量的输入，同是输出也是张量，下图为上面生成w的计算图，其中(weight)代表weight，它由weight_tensor通过assign操作生成，同时它也能够通过read输出tensor<br>

![variable and tensor](img/tensor_variable.png)

在声明Variable时有一个很重要的参数`trainable`，只有trainable设置为true的变量才会被TensorFlow在迭代时进行更新。

## 5. 使用TensorFlow构建一个简单的手写数字识别程序

&emsp;&emsp;在这个示例中，将会使用TensorFlow构建一个基本的线性分类模型来对[mnist数据集](http://yann.lecun.com/exdb/mnist/)中的的手写数字进行识别。MNIST数据集是一个经典的分类数据集，它的每个样本是一个28\*28的灰度图，每个图片展示的是手写数字0~9中的一个，如下图示。

![mnist dataset](img/MNIST.png)

任务目标就是识别数字，即将数字正确的分类，采用模型的一些参数如下：

+ 模型：softmax回归，即逻辑回归在多分类情况下的推广形式，$p(y=i)=\frac{e^{-w_ix}}{\sum_j e^-w_jx}$

+ 特征：图片的灰度值特征

+ 损失函数：log损失函数 $loss = -\sum_i I_{i=lable}logp(y=i)$

下面是代码示例:
>这些代码可以在我们提供的jupyter notebook示例[tensorflow example](src/linear_model.ipynb)中直接运行查看。

```python
import tensorflow as tf
import time

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 预处理
train_images = x_train.reshape(-1, 28*28).astype('float32') / 255.0
test_images = x_test.reshape(-1, 28*28).astype('float32') / 255.0
train_labels = tf.one_hot(y_train, 10).numpy()  # 转换为one-hot编码
test_labels = tf.one_hot(y_test, 10).numpy()

# 参量设置
batch_size = 128
max_steps = 10000
step_size = 0.001

# 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, 
                         activation='softmax',
                         kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                         bias_initializer='zeros')
])

# 定义损失函数和优化器
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=step_size)

# 创建数据集
train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_images, train_labels)).shuffle(60000).batch(batch_size)
train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_images, train_labels)).shuffle(60000).batch(batch_size).repeat()
# 测试数据预处理
test_data = (test_images, test_labels)

# 训练循环
start_time = time.time()

for step, (x_batch, y_batch) in enumerate(train_dataset.take(max_steps)):
    with tf.GradientTape() as tape:
        predictions = model(x_batch, training=True)
        loss_value = loss_fn(y_batch, predictions)
    
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    if step % 50 == 0:
        test_pred = model.predict(test_images, batch_size=batch_size, verbose=0)
        correct_prediction = tf.equal(tf.argmax(test_pred, 1), tf.argmax(test_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        time_cost = (time.time() - start_time) * 1000 / 50
        start_time = time.time()
        
        print(f"Step: {step:5d}, Test accuracy: {accuracy:.3f}, timeCos: {time_cost:.1f}ms")

# 最终评估
final_pred = model.predict(test_images, verbose=0)
final_accuracy = tf.reduce_mean(tf.cast(
    tf.equal(tf.argmax(final_pred, 1), tf.argmax(test_labels, 1)), 
    tf.float32))
print(f"\nFinal test accuracy: {final_accuracy:.3f}")
```

运行10000次后，模型在测试集上的分类正确率达到了0.85。

