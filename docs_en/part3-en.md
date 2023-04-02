<p align="center">
  <img src="https://cdn.discordapp.com/attachments/724700045525647420/729135226365804594/SFNNUE2.png">
</p>

<!-- <h1 align="center">国际象棋程序Stockfish NNUE设计简介（三）网络结构</h1> -->
<h1 align="center">A Brief Introduction to Stockfish NNUE: (3) Network Architecture</h1>

<!-- 
由于要限制向前传播的计算时间，NNUE的网络结构并不复杂，是典型的全连接神经网络，总共只有四层。其中第一层（输入层）参数数量非常巨大，后三层则相对较小，呈现“头大身子小”的形状。下面的图是Nodchip制作的关于HalfKP_256X2_32_32的网络结构图。

| ![title](./img/p3-1.png) |
| :---: |
| <em>Nodchip制作的HalfKP_256X2_32_32网络结构示意图</em> |


其中最左侧棋盘表示部分光看图的话有些不好理解。借用上一篇提到的概念，"active"指己方，"opponent"指对方。"King, Piece - Squares"其实是King Square - Piece Square，也就是“王所在格子”和“其他棋子所在格子”的组合。我在上一篇中将此组合定义为了“位置关系”。图上剩下的部分就是网络结构。 在正式的介绍网络结构之前，先介绍一下NNUE网络中用到的各种子结构，相当于PyTorch中的nn.Module。

**AfflineTransformer Layer:** 也就是PyTorch里的nn.Linear，就是最简单的$y=wx+b$。AfflineTransformer(x, dinput, doutput, type)表示该层的输入为x，输入维度为dinput，输出维度为doutput，以及参数类型是type。

**ClippedReLu Layer:** 不同于原版ReLu，ClippedReLu不但有下界，还有一个上界。所有小于下届（lowerbound）或大于上界的（upperbound）的输入都会分别被替换为下界或上界。

```
 ClippedReLu(x, lowerbound, upperbound) = max(lowerbound, min(x, upperbound))
```

在NNUE的实现中，lowerbound = 0， upperbound = 127（int8的最大正数取值）。NNUE网络使用ClippedRelu是因为NNUE网络的参数和计算结果均以整型存储，而C/C++中，整型的上溢或下溢都是不会报错的。而127的上界又很容易越过，任何大于整型上界的变量值在赋给8位整形变量时，经过类型强制转换后会得到难以预料的非法数值。为了防止这些数值的出现，ClippedReLu 直接超过上界的输入“裁剪”为上界。

**ScaledClippedReLu Layer:** ScaledClippedReLu比一般的ClippedReLu多了一步“放缩”，即将输入除以一个分母，先将其缩小数倍，再输入到ClippedReLu得到输出。

```
ScaledClippedReLu(x, lowerbound, upperbound, denominator) 
    = ClippedReLu(x / denominator, lowerbound, upperbound)
```
其中denominator是缩小的倍数。使用放缩是因为NNUE的网络的计算结果要以8位整型存储，正数上界为127。而经过矩阵法乘法加法运算后，每个维度的输出范围的上界会比127大得多。如果不进行放缩，很多超过127的输出无论多大都会被裁剪为127，这就使得原本大小有区别的输出变得没有了区别，损失了很多信息。经过放缩之后信息就得到了一定程度的保留，尽管粒度变得更粗糙了。在实际应用中，denominator = 64，并且除法是通过向右位移6位实现的。

实际上，包括“裁剪”、“防缩”在内的这些额外处理大都是由使用整型参数引起的。而使用整型参数的目就是为了利用CPU中的高级指令来并行化的实现小规模的向量乘法与加法。我会在后文中解释具体的做法。

按照NNUE的实现，这个网络可以分成两部分：第一层被又被称作FeatureTransformer，剩余的部分被称为Network。
-->

Since we need to limit the computation time of forward propagation, the network architecture of NNUE can not be super complicated. It is a typical fully connected neural network (FCN) with only four layers in total. Among them, the number of parameters in the first layer (input layer) is much larger than the following three layers, which is showing a shape of "big-head-small-body". The figure below is the network architecture diagram of HalfKP_256X2_32_32 by Nodchip.

| ![p3-1](./img/p3-1.png) |
| :---: |
| <em>Diagram of HalfKP_256X2_32_32 network architecture created by Nodchip.</em> |

It is probably a little difficult to understand the board representation by the left most chess board picture in this diagram. 
Referring to the concepts mentioned in the previous article, "active" refers to "our side", and "opponent" refers to "their side". "King, Piece - Squares" is actually a short of "King Square - Piece Square", which is a combination of "the square where the king is" and "the square where other pieces are". I defined this combination as "piece-square relationship" in the previous article. The rest of the diagram is about network architecture. Before formally introducing the network architecture, we need to first introduce the different layers used in the NNUE network, which are equivalent to `nn.Module` in PyTorch.

**AfflineTransformer Layer:** Corresponds to `nn.Linear` in PyTorch, which is simply $y=wx+b$. AfflineTransformer(x, dinput, dooutput, type) indicates that the input of this layer is `x`, the input dimension is `dinput`, the output dimension is `dooutput`, and the parameters are in `type` (which are usually integer types in NNUE).

**ClippedReLu Layer:** Different from the original ReLu, ClippedReLu does not only have a lower bound, but also an upper bound. All inputs less than the lower bound or greater than the upper bound are replaced by the lower bound or upper bound, respectively.

```
ClippedReLu(x, lowerbound, upperbound) = max(lowerbound, min(x, upperbound))
```

In the implementation of NNUE, lowerbound = 0, upperbound = 127 (the maximum positive value of int8). The NNUE network uses ClippedRelu because the parameters and calculation results of the NNUE network are stored in integers, and in C/C++, integer overflow or underflow will not report errors. And the upper bound of 127 is easy to cross. When any variable value greater than the upper bound of the integer is assigned to an 8-bit integer variable, an unpredictable illegal value will be obtained after type conversion. To prevent these cases, ClippedReLu “clips” inputs to the upper bound.

**ScaledClippedReLu Layer:** ScaledClippedReLu has one more step of "scaling" than the general ClippedReLu, which is to divide the input by a denominator, first reduce it several times, and then input it to ClippedReLu to get the output.

```
ScaledClippedReLu(x, lowerbound, upperbound, denominator) 
    = ClippedReLu(x / denominator, lowerbound, upperbound)
```
where `denominator` is the scaling multiplier. 
Scaling is used because the computation results of the NNUE network are stored in 8-bit integers, and the upper limit of positive numbers is 127. After matrix multiplication and addition, the upper bound of the output range of each dimension will be much larger than 127. If scaling is not performed, many outputs exceeding 127 will be cropped to 127 no matter how large they are, which makes the output with different sizes indistinguishable and therefore loses a lot of information. After scaling, the information can mostly be preserved, although the granularity becomes coarser. In practice, denominator = 64, and division is implemented by doing right-shift for 6 bits.

In fact, most of these extra processing including "cropping" and "shrinkage" are caused by using integer parameters. The purpose of using integer parameters is to realize small-scale vector multiplication and addition in parallel using Intel advanced SIMD CPU instructions. I will explain how to do that in the later sections.

According to the implementation of NNUE, the network contains two major parts: the first layer which is also called "FeatureTransformer" and the remaining three-layer part which is called "Network".


<!-- 
### FeatureTransformer部分

用上文提到的层来描述的话，FeatureTransformer的结构如下：

AfflineTransformer(x, 41024, 256, int16) -> ClippedReLu(x, 0, 127)

以上的变换只考虑了己方或对方当中的某一方。实际上双方的棋盘表示都需要经过此变换。因此对任意一个己方和对方的棋盘表示分别为x1和x2的局面，FeatureTransformer中的变换严格来说是如下进行的：

* AfflineTransformer(x1, 41024, 256, int16) -> ClippedReLu(x1', 0, 127) -> y1
* AfflineTransformer(x2, 41024, 256, int16) -> ClippedReLu(x2', 0, 127) -> y2
* y = y1 $\oplus$ y2

注意以上计算y1和y2用到的参数是一样的，只是输入有两个，因此输出也是两个：y1与y2，各自256维。按顺序拼接在一起后，就得到512维的输出y。


| ![title](./img/p3-2.png) |
| :---: |
| <em>FeatureTransformer部分网络结构示意图</em> |

以上的层只是为了准确的描述。实际上第一层的输出并不是在估值时才通过调用向前传播来计算，而是也和棋盘表示一起随着棋局着法执行和撤销增量进行的。NNUE将第一层单独拿出来也是为便于进行这种增量计算。

如上篇所述，某一方的棋盘表示x是一个二进制的稀疏向量，其非零元素下标集合为$\{i_1, i_2, ..., i_k\}$。我们假设x是一个41024维的横向量，那么对应的，AfflineTransformer 中的b也是一个256维的横向量，w的形状为41024 x 256。我们可以把w的每一行看作一个向量，于是w可以表示为41024个256维的行向量组成的向量数组， 第$i$行向量用$w_i$表示。那么，wx+b就可以简化为一串向量累加运算：$wx+b = \sum^k_{j}w_{i_j} + b$。这一过程和我们在NLP中使用word embedding矩阵计算一个句子（假设句子encoding是由其包含的所有不重复词的one hot encoding相加得来的）的embedding的过程极为相似。

| ![title](./img/p3-3.png) |
| :---: |
| <em>稀疏二进制向量x与w做乘法可以简化为挑选w中x的非零元素所对应的行的和。类比NLP中的例子，如果忽略b，并假设x是一个词的one hot encoding，那么这个过程就是在计算x的wording embedding</em> |

随着对局的进行，x非零元素下标集合也在增量的改变，对应的以上累加结果也可增量的改变。可以设想一下，如果每个$w_i$是一个数字的话，这个累加的更新过程就变的极其简单快速了：只要减掉从集合中消失了的下标对应的$w_i$，再加上新出现的$w_j$就可以了。其实换成向量，过程也类似，只是数字的加减变成了若干256维的整型向量的加减法。如果向量的加减也能像数字加减一样，那一次向前传播的计算速度就会大大加快。而这正是NNUE使用整型参数的最终目的：利用高级CPU指令实现快速向量加减法。实际实验证明，即使仅使用SSE4.2指令集的CPU上，Stockfish-NNUE的搜索速度也比用使用for循环实现向量加减法的版本快了近一倍，这将对棋力产生重大影响。而使用最新的AVX2或AVX512指令集的Stockfish-NNUE将会更快。
-->



### FeatureTransformer Part

With the layers mentioned above, the architecture of FeatureTransformer can be described as follows:

AfflineTransformer(x, 41024, 256, int16) -> ClippedReLu(x, 0, 127)

The above transformation only considers one of "our side" or "their side". 
In fact, this transformation needs to be taken on the board representation vector of both sides. Therefore, for any position's our side and their side, denoted by x1 and x2 respectively, the transformation in FeatureTransformer is done as follows:

* AfflineTransformer(x1, 41024, 256, int16) -> ClippedReLu(x1', 0, 127) -> y1
* AfflineTransformer(x2, 41024, 256, int16) -> ClippedReLu(x2', 0, 127) -> y2
* y = y1 $\oplus$ y2

Note that the parameters used to compute y1 and y2 above are the same. But since there are two inputs, we get two outputs as well: y1 and y2, each with 256 dimensions. After concatenating these two outputs together in order, a 512-dimensional output y is obtained.

| ![3-2](./img/p3-2-en.png) |
| :---: |
| <em>Diagram of FeatureTransformer part network achitecture</em> |

The above layers are for the convenience of describing only. In fact, the output of the first layer is not calculated by calling forward propagation at the time of evaluation, but is incrementally updated along with the board representation as the game is played. NNUE takes out the first layer separately to facilitate this incremental computation.

As mentioned in the previous article, the board representation x of any side is a binary sparse vector, and its index set of non-zero elements $A = \{i_1, i_2, ..., i_k\}$. We assume that x is a 41024-dimensional row vector, then correspondingly, b in AfflineTransformer is also a 256-dimensional row vector, and the shape of w is 41024 x 256. If we view w in the row-major perspective, w can be seen as a vector array consisting of 41024 256-dimensional row vectors, where the $i$th row vector can be denoted by $w_i$. 
Then, wx+b can be simplified as a series of vector accumulation operations: $wx+b = \sum^k_{j}w_{i_j} + b$, where $i_j \in A$. This process is very similar to the process when we use the word embedding matrix to compute the embedding of a sentence (assuming that the sentence encoding is obtained by adding the one-hot encoding of all the non-repetitive words it contains) in NLP.

| ![3-3](./img/p3-3.png) |
| :---: |
| <em>The multiplication of a sparse binary vector x by w can be simplified as picking the sum of rows corresponding to the non-zero elements of x in w. Analogous to the example in NLP, if we ignore b and assume x to be the one-hot encoding of a word, then this process is to compute the word embedding of x.
</em> |

As the game is ongoing, the index set of non-zero elements of x is also incrementally changed, and the corresponding above accumulation results can also be incrementally updated.
Imagine that if each $w_i$ is a number where $i$ is from a index set $A$, and we want to incrementally maintain $\sum_{i \in A} w_i$, the cumulative update process would be pretty simple and fast: everytime after $A$ changed, we just subtract the $w_i$ from the old sum for each index $i$ that disappeared from $A$ and add the $w_j$ to the old sum for each index j that is newly inserted into $A$, then the new sum is obtained.
Now let's change the type of $w_i$ from a number to a vector. The process above is still the same, except that the “addition and subtraction of numbers" becomes the "addition and subtraction of several 256-dimensional integer vectors". 
**If the addition and subtraction of vectors can be done like addition and subtraction of numbers in our program, the computation speed of that forward propagation will be significantly accelerated.**
This is actually the ultimate purpose of NNUE using integer parameters: to use advanced CPU SIMD instructions to achieve fast vector addition and subtraction. 
Experiments have proved that even on a CPU that only uses the SSE4.2 instruction set, the search speed of Stockfish-NNUE can be nearly twice as fast as the version that uses for-loops to implement vector addition and subtraction, which will have a significant impact on engine's ELO.
Stockfish-NNUE using the latest AVX2 or AVX512 instruction set can run even faster.



<!-- 
### Network部分

严格来说，Network在代码中并不是一个类。NNUE将多层的全连接网络存成了一个链表，链表中每个元素是一层。每层只记录自己的上一层，且都有自己向前传播函数。每一层在计算向前传播时，首先计算上一层的向前传播，并将其输出作为本层输入，再计算自己的输出。于是，当调用链表尾端最后一层的向前传播，通过层层递归，就相当于计算了整个网络的向前传播。NNUE其实只是把最后一层“重命名”为了Network。用上文提到的层来描述的话，Network部分的结构如下：

AfflineTransformer(x, 512, 32, int8) -> ScaledClippedReLu(x, 0, 127, 64) -> AfflineTransformer(x, 32, 32, int8) -> ScaledClippedReLu(x, 0, 127, 64) -> AfflineTransformer(x, 32, 1, int8)

| ![title](./img/p3-4.png) |
| :---: |
| <em>Network部分网络结构示意图</em> |

与FeatureTransformer类似，Network部分中的矩阵乘法也可以通过快速向量操作来实现。不过这里的输入向量不再是二进制稀疏向量，因此需要实打实的计算输入行向量与w的列向量的点积，再求和。其中点积的部分也可以通过CPU指令优化。好在这部分的网络规模远远小于输入层，做密集矩阵的计算也不会额外耗时太多。

![title](./img/p3-5.png)
-->

### Network Part

Technically, Network is not a class in Nodchip's code. NNUE stores the fully connected network as a linked list, where each element is a layer. Each layer only needs to remember its previous layer (or "parent" layer in terms of linked list), and implement its own forward propagation function. When each layer computes its forward propagation, it first computes the forward propagation of its parent layer, and uses the parent layer output as the input of the current layer to compute its own output. Therefore, when the forward function of the last layer at the tail of that linked list is called, through layer-by-layer recursion, it is equivalent to computing the forward propagation of the entire network. NNUE actually simply renames the last layer to "Network". The architecture of the Network part is as follows:

AfflineTransformer(x, 512, 32, int8) -> ScaledClippedReLu(x, 0, 127, 64) -> AfflineTransformer(x, 32, 32, int8) -> ScaledClippedReLu(x, 0, 127, 64) -> AfflineTransformer( x, 32, 1, int8)

| ![p3-4](./img/p3-4-en.png) |
| :---: |
| <em>Diagram of the architecture of the Network part</em> |

Similar to FeatureTransformer, matrix multiplication in the Network part can also be implemented with fast vector operations. However, the input vector here is no longer a binary sparse vector, so it is necessary to actually compute the dot-product of the input row vector and the column vector of w, and then do the sum. The part of the dot-product can also be optimized by CPU instructions. Fortunately, the network size of this part is much smaller than the input layer, and the computation of the dense matrix will not take too much extra time.

![p3-5](./img/p3-5.png)



<!-- 
### 网络输出与最终估值输出

注意上面的网络的最后一层是没有activation的，而是直接把最后的线性变换的结果输出了，因此这个输出的范围比activation的输出范围要大的多。在不考虑bias的情况下，最后一层输出的可以在区间 [-127 x 128 x 32, 127 x 127 x 32]。加上bias之后值域会有偏移，但范围的大小不变。 然而这个范围相对于Stockfish的估值还是太大了。Stockfish的手写估值函数返回值是一个16位整型值，范围 [-32000, +32000] （Stockfish将绝对值超过这个范围一些值用作了特殊值，比如“非法”、“困毙”、“在n步内困毙”）。为了将网络的输出映射到Stockfish的传统估值范围，NNUE将网络输出做了最后一步变换：

y = clip((x / 16), -32000, 32000)

其中，clip函数类似ClippedRelu层的变换，将超出上界或下界的输入值替换为上界或下界。从上一部分的描述可以看到，Network最后一层输出范围是127*127*32 - (-127*128*32) = 1036320，恰好是一个20位整型变量的取值范围。除以16相当于右移4位，结果正好是一个16位整数，范围[-32768，+32767]。裁剪掉绝对值超过32000的部分，就得到了一个合法的Stockfish局面估值。

为了避免语言描述的不准确，我用Pytorch实现了一下以上描述的NNUE网络（不包含最终估值输出的变换），代码详见[Github](https://github.com/nkg114mc/sfnnue-intro/blob/master/pytorch-nnue-net.py)[^1]。
-->


### Network Output and Final Evaluation Score

Note that the last layer of the above network does not have activation, but directly outputs the result of the final linear transformation, so the value range of this output is much larger than the activation output range. Without considering the bias, the output of the last layer can be in the interval [-127 x 128 x 32, 127 x 127 x 32]. After adding bias, the value range will be offset, but the size of the range will not change. However, this range is still too large compared to expected Stockfish evaluation scores. The return value of hand-written Stockfish evaluation function is a 16-bit integer value with a range of [-32000, +32000] (Stockfish uses some values that are out of this range as special values, such as "None", "Infinite", "Mate", "Mate in N steps", etc.). In order to map the network output to the classical evaluation range of Stockfish, NNUE makes the last transformation of network output as follow:

y = clip((x / 16), -32000, 32000)

Where the clip function is similar to the transformation of the ClippedRelu layer: replacing the input value beyond the upper or lower bound with the upper or lower bound. 
As can be seen from the previous sections, the output range of the last layer of Network is 127*127*32 - (-127*128*32) = 1036320, which is exactly within the range of a 20-bit integer variable. Dividing by 16 is equivalent to right-shifting by 4 bits, and the result is exactly a 16-bit integer with a range of [-32768, +32767]. Clipping off the parts with an absolute value over 32000 will give a valid stockfish position evaluation.
To avoid inaccuracy of my description, I used Pytorch to implement the NNUE network described above (not including the transformation of the final output). The code is available in [Github](https://github.com/nkg114mc/sfnnue-intro/blob/master/pytorch-nnue-net.py)[^1].


---


<!-- 
下面的内容是关于NNUE网络及其向前传播的具体实现，属于源代码的细节，如果只是对NNUE理论感兴趣的童鞋可以直接忽略。

### nn.bin文件结构

Stockfish的测试系统FishTest专门创建了一个页面[NN stats](https://tests.stockfishchess.org/nns)，用于用户上传他们训练出来的权重。如果你曾经从上面的网页下载过权重，就会注意到这个二进制文件有大约21MB。这里就简单介绍一下这个文件的结构。nn.bin文件按照字节流的顺序，存储了以下信息：

#### Header部分

* **version**: uint32，4字节。版本信息，为将来升级而预留的信息。
* **hashvalue**: uint32，4字节。Header的hashvalue是FeatureTransformer和Network部分各自的hashvalue异或之后的结果。
* **arch_name_size**: uint32，4字节。arch_name是一个以字节流存储的字符串，包含两部分：长度和内容。这里首先是4字节的长度信息，表示这个字符串包含的字符个数。每个字符一字节。
* **arch_name_string**: char[]，<arch_name_size>字节。就是字符串的内容。例如，HalfKP_256X2_32_32结构会输出如下内容，其实就是一个人类可读的网络结构描述。

```
Features=HalfKP(Friend)[41024->256x2],Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-32](ClippedReLU[32](AffineTransform[32<-512](InputSlice[512(0:512)])))))
```

#### FeatureTransformer部分

* **hashvalue**: uint32，4字节。hashvalue是由输入维度，输出维度等重要参数为输入计算出的32位哈希值，以保证读入权重的网络结构与Stockfish-NNUE正在使用的网络结构是一致的。
* **bias**: int16[]，256 * 2字节。存储输入层的b向量。
* **weights**: int16[]，41024 * 256 * 2字节。存储输入层的w矩阵。w本来是一个二维数组，这里NNUE将它拉平成为了一个一维数组。拉平的方法是先行后列，也就是保证行向量元素的存储空间依旧连续。

#### Network部分

* **hashvalue**: uint32，4字节。与FeatureTransformer的hashvalue类似。这里的hashvalue是之后三层各自计算出的hashvalue异或在一起之后的结果。
* **bias**: int8，32 * 1字节。存储第二层的b向量。
* **weights**: int8[]，512 * 32 * 1字节。存储第二层的w矩阵。注意这里的w的拉平方法与FeatureTransformer正好相反，是先列后行，保证每个列向量元素的存储空间依旧连续。
* **bias**: int8[]，32 * 1字节。存储第三层的b向量。
* **weights**: int8，32 * 32 * 1字节。存储第三层的w矩阵，先列后行。
* **bias**: int8[]，1 * 1字节。存储第四层的b向量。
* **weights**: int8[]，32 * 1 * 1字节。存储第四层的w矩阵，先列后行。

注意，两部分对w的存储有所不同：FeatureTransformer中的w是按照行向量优先存储的，而其他层的w则是按照列向量优先存储。初次遇到这种不一致可能会很费解，然而如果搞明白了NNUE是如何做矩阵乘法的，就不难理解了。因为在FeatureTransformer部分，矩阵乘法计算主要以w部分的行向量相加来进行，因此保证行向量优先便于成块的读取内存中w的行向量来作为CPU向量指令的操作数。相反，在Network部分，矩阵乘法则要先计算输入向量与w列向量的点积，因此需要成块读取的操作数就变成了列向量。这一切都是为了CPU的向量指令能够方便的从内存中读取操作数而设计的。

| ![title](./img/p3-6.png) |
| :---: |
| <em>w矩阵拉平方法：(a)为FeatureTransformer使用的“先行后列”，(b)为其他部分使用的“先列后行”</em> |
-->


The following content is about the implementation of NNUE network and its forward propagation. It relates to the details of the source code. So if you are only interested in the general mechanism of NNUE but have no interest to the code-level details, you can directly skip the rest of this section.

### nn.bin File Structure

Stockfish test system FishTest has created a page [NN stats](https://tests.stockfishchess.org/nns) specifically for users to upload their trained weights on their site. If you have ever downloaded weights from the above site, you'll notice that the binary file (usually with a name like "nn-xxxxxxxxxxxx.bin") is about 21MB. Here is a brief introduction to the structure of this file. The `nn.bin` file stores the following information in the order of the byte stream:

### Header Part

* **version**: uint32, 4 bytes. Version information, information reserved for compatibility check when future upgrade happens.
* **hashvalue**: uint32, 4 bytes. The hashvalue of the Header is the XOR result of the hashvalues ​​of the FeatureTransformer and Network parts respectively.
* **arch_name_size**: uint32, 4 bytes. arch_name is a string stored as a byte stream, consisting of two parts: length and content. Here first is the length information of 4 bytes, indicating the number of characters contained in this string. One byte per character.
* **arch_name_string**: char[], <arch_name_size> bytes. the actual string content. For example, the `HalfKP_256X2_32_32` architecture will output the following string (which is just a human-readable description of the network architecture):

```
Features=HalfKP(Friend)[41024->256x2],Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-32](ClippedReLU[32](AffineTransform[32<-512](InputSlice[512(0:512)])))))
```

### FeatureTransformer Part

* **hashvalue**: uint32, 4 bytes. hashvalue is a 32-bit hash value computed from all important properties such as input dimension and output dimension, to ensure that read weight is compatible with the network architecture that Stockfish-NNUE is currently using.
* **bias**: int16[], 256 * 2 bytes. Stores the b vector of the input layer.
* **weights**: int16[], 41024 * 256 * 2 bytes. Stores the w matrix of the input layer. w was originally a two-dimensional array, here NNUE flattens it into a one-dimensional one. The order of doing flattening is row-major (first row then column), that is, to ensure that the row vector elements are still stored continuously.

### Network Part

* **hashvalue**: uint32, 4 bytes. Similar to the hashvalue of FeatureTransformer. The hashvalue  is the XOR result of all the hashvalues ​​calculated by each of the next three layers.
* **bias**: int8, 32 * 1 byte. Store the b vector for the second layer.
* **weights**: int8[], 512*32*1 bytes. Stores the w matrix of the second layer. Note that the flattening order of w here is the opposite of order in FeatureTransformer, which is column-major (column first and then row), ensuring that each column vector elements are stored continuously.
* **bias**: int8[], 32 * 1 byte. Store the b vector for the third layer.
* **weights**: int8, 32*32*1 bytes. Store the w matrix of the third layer, column-major order.
* **bias**: int8[], 1 * 1 byte. Store the b vector for the fourth layer.
* **weights**: int8[], 32*1*1 bytes. Store the w matrix of the fourth layer, column-major order.


Note that the storage of w in the two parts is different: w in FeatureTransformer is stored in a row-major order, while w in Network layers are stored  column-major order. It might be confusing to see this inconsistency at first glance, but if you understand how NNUE does matrix multiplication, there would be nothing mysterious. 
In FeatureTransformer, the matrix multiplication computation is mainly performed by adding the row vectors of w, thus we store w in a row-major order to ensure that the row vectors of w can be loaded from memory in blocks as the operands of the CPU vector instructions.
On the contrary, in the Network part, since all w matrices are dense matrices, matrix multiplication has to follow the standard steps: first do the dot-product of the input (row) vector and each w column vector and second compute the sum. 
As a result, the operands that need to be loaded in blocks now become column vectors, which makes w to be stored in the column-major order in memory.
All of these are designed for the CPU vector instructions to conveniently load operands from memory.

| ![title](./img/p3-6.png) |
| :---: |
| <em>$w$ matrix flattening orders: (a) "row-major" order applied in FeatureTransformer, (b) "column-major" order applied in other layers.</em> |



<!-- 
### 使用CPU SIMD指令的快速整数向量操作

如上文所述，我们希望能够像计算两个数字相加一样快速的计算两个向量的相加。以FeatureTransformer中的w为例：每一个行向量  是一个长度为256类型为int16的数组。如果用最一般的C语言实现这个加法，那么只能是通过for循环对每个数组元素做256次加法。这个过程有256次操作，显然会比较慢。如果存在一个指令，仅通过一个操作就能实现这个数组的加法，那无疑是最完美的。

走运的是，CPU中存在一些高级指令能够允许用户能实现一定限度的数据层面的并行化。这种技术又被称为单指令流多数据流（SIMD）。例如在英特尔CPU中的AVX2指令集，就允许对两个256位的向量（相当于C/C++中的数组，就是32字节的连续内存块）直接进行一些二元操作。用户可以定义向量中每个元素占用的字节数。例如，我可以把这256位的向量看作长度为32的int8向量，也可以是长度为16的int16向量等等。最大的类型可以到int64。两个向量的加法可以通过专用的函数_mm256_add_epi16来完成。此外还可以进行减法，乘法，赋值，归零等等操作。

回到最开始FeatureTransformer的例子，每个$w_i$是256 x 2 = 512字节（int16是2字节）的向量。在AVX2指令集中，一次可以计算256位也就是32字节的向量的加法。那么两个  的加法只需要for循环16次这样的操作就可以完成了。相比于最原始的for循环，我们的操作个数只有原来的1/16，速度大大加快。类似的SSE3指令集也允许这种向量操作，只不过它的最大带宽只有128位或是16字节。

其实，以上的操作在GPU计算中非常常见，而且规模要大得多。NNUE的作者使用SIMD指令做整数向量操作，某种意义上可以视为使用了“集成在CPU中的微型GPU”来实现了向量运算。我个人并不清楚作者这样做的明确目的。猜测大概是为了避免对GPU的硬件依赖，毕竟不是每台计算设备都配备有高性能的GPU。


### 补充：关于NNUE不使用GPU的原因

感谢 @恨铁不成钢琴 的补充。关于NNUE不使用GPU的原因，我觉的他说的更有道理：

> SIMD的用法是减少计算延迟用的。在剪枝的时候，剪枝算法的batch_size并不会太大（传统剪枝的batch_size其实是1，因为每次只针对一个局面判断是否剪枝）。如果此时使用GPU，会带来很大的延迟（传数据去GPU，剪枝，再传回来，延迟并不可以忽略）。现在鳕鱼干NNUE搜索的速度大约是每秒几千万个节点，GPU很难在每秒处理几千万个节点，如果真的用神经网络，只要是块nvidia的显卡计算速度都比（同时代）cpu快10倍左右——但考虑到通信成本，我们并不能完美利用GPU的计算速度。

由于alpha-beta搜索的序列性（即子节点要按顺序搜索，因为不知道哪个会发生beta剪枝），NNUE的向前传播需要在batch_size为1的条件下进行。CPU和GPU通信是有延迟的。当batch_size较大时，GPU带来的加速完全可以抵消这种延迟。然而当batch_size = 1时，延迟的成本完全盖过了GPU加速的收益，得不偿失。
-->


### Fast Integer-Vector Operations with CPU SIMD Instructions

As mentioned above, we hope to add two vectors as fast as adding two numbers. Take w in FeatureTransformer as an example: each row vector is an array of length 256 and type int16. If this addition is implemented in the most general C language, then only 256 additions can be made to each array element through a for loop. This process has 256 operations, which will obviously be slower. 
If there is an instruction that can realize the addition of this array through only one operation, it is undoubtedly the most perfect.

Fortunately, there are some advanced instructions in the CPU that allow the user to apply some data-level parallelization. This technique is also known as Single Instruction Multiple Data (SIMD). For example, the AVX2 instruction set in the Intel CPU allows some binary operations to be performed directly on two 256-bit vectors (equivalent to an array in C/C++, which is a 32-byte continuous memory block). The user can define the number of bytes each element can take by selecting the element type. For example, this 256-bit vector can be interpreted to an int8 vector of length 32, or an int16 vector of length 16, etc. The largest type can be up to int64. The addition of two vectors can be done by the dedicated function `_mm256_add_epi16`. Some other operations such as subtraction, multiplication, assignment, and zeroing are also included.

Going back to the original FeatureTransformer example, each $w_i$ is a vector of 256 x 2 = 512 bytes (int16 takes 2 bytes). 
In the AVX2 instruction set, addition operation can take two 256-bit operands, which is 32 bytes, at one time. 
Then the addition of two $w_i$ only takes 16 steps of for-loop to complete (512 / 32 = 16). 
Compared with the most naive for-loop, the number of required operations is only 1/16 of the original, and the computation gets a great speedup. The SSE3 instruction set also allows this kind of vector operations, but its maximum bandwidth for operrands is only 128 bits or 16 bytes.

<!-- 
In fact, the above operations are very common in GPU computing, and the scale is much larger. The author of NNUE uses SIMD instructions to perform integer vector operations. In a sense, it can be regarded as using a "miniature GPU integrated in the CPU" to implement vector operations. I personally don't know the author's explicit purpose for doing this. The guess is probably to avoid hardware dependence on the GPU, after all, not every computing device is equipped with a high-performance GPU. 
-->

### Supplement: About the Reason Why NNUE didn't Use GPU

Thanks @恨铁不成琴琴 for the supplement. Regarding the reason why NNUE does not use GPU, I think what he said is reasonable:

> “The usage of SIMD is to reduce computation overhead. When doing pruning during search, the batch_size of the pruning algorithm will not be too large (the batch_size of traditional pruning is actually 1, because it only checks one position each time about whether it should be pruned or not).
Using the GPU in this situation will bring a lot of computation overhead (transmitting data to the GPU, pruning, and then sending it back. The overhead time cannot be ignored). Nowadays the search speed of Stockfish NNUE is about millions of nodes per second. It would be difficult for GPUs to process millions of nodes per second. If the neural network was applied in evaluation, given any Nvidia graphics card, the (forward propagation) computational speed could be 10 times faster than any (contemporary) CPU - but considering the communication cost, we can not perfectly utilize the GPU's computing speed.”

Due to the sequential nature of alpha-beta search (child nodes have to be searched in sequential order, because no one at which child node that a beta-cut may happen), the forward propagation of NNUE needs to be run with the batch_size = 1. There is communication overhead when passing data from CPU to GPU. When the batch_size is large, the acceleration brought by the GPU can completely overrun this overhead time. However, when batch_size = 1, the cost of communication overhead would completely cancel out the benefit of GPU acceleration.


<!-- 
### 参考

[^1]: 该网络结构是2020年Stockfish-NNUE最初发布时的版本。此后网络结构由经过了几次更新，如金已经和本文描述的不太一样了，请参考nnue-pytorch项目的文档以了解最新的网络结构。
-->

### References

[^1]: The network architecture is the version in 2020 when Stockfish-NNUE was first released. Since then, the network architecture has been updated several times, and now it is NOT the same as described in this article anymore. Please refer to the documentation of the nnue-pytorch project to learn about [the latest network architecture](https://github.com/glinscott/nnue-pytorch/blob/master/docs/nnue.md#architectures-and-new-directions).
