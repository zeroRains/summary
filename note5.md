# 第7.8周学习报告  

`@Author 卢林军`  
`@Date 2020415`  
[Mini-batch梯度下降法](#1) | [动量梯度下降（momentum梯度下降法）](#2) | [RMSprop](#3) | [Adam优化算法](#4) | [学习率衰减](#5) | [](#6)

```
优化算法
```

# <a id='1'>Mini-batch梯度下降法</a>

1. 把训练集分割为小一点的子训练集，这些自己被取名为Mini-batch
2. 假设我们的总样本数有5000000，每个Mini-batch有1000个样本，那么我们的数据集合标签可以划分成5000个Mini-batch（1000*5000=5000000）
3. 我们把在数据集中划分出来的Mini-batch称为$X^{\lbrace 1 \rbrace},X^{\lbrace 2 \rbrace}...X^{\lbrace n \rbrace}$,由他们组成新的X
4. 我们把在标签中划分出来的Mini-batch称为$Y^{\lbrace 1 \rbrace},Y^{\lbrace 2 \rbrace}...Y^{\lbrace n \rbrace}$,由他们组成新的Y
5. 因此我们把第t个mini-batch称作：$X^{\lbrace t \rbrace},Y^{\lbrace t \rbrace}$
6. 回顾一下这些符号
    * $x^{(i)}$是第i个训练样本
    * $z^{[l]}$表示神经网络中第l层的Z值
    * $X^{\lbrace t \rbrace},Y^{\lbrace t \rbrace}$表示不同的mini-batch
7. mini-batch的维度：
    * n代表特征数，m代表每个mini_batch划分出来的样本数
    * 对于$X^{\lbrace t \rbrace}$他的维度是（n,m）
    * 对于$Y^{\lbrace t \rbrace}$他的维度是（1,m）
8. 算法思路：
    * 假设我们有5,000,000个样本，我们每个mini-batch中有1000个数据
    * 用for循环遍历5000次，每次都代表一个$X^{\lbrace t \rbrace},Y^{\lbrace t \rbrace}$
    * 然后用他们完成向前传播
    * $Z^{[l]} = W^{[l]}X^{\lbrace t \rbrace} + b$
    * $A^{[l]} = g^{[l]}(Z^{[l]})$
    * 计算cost值,这里的x,y都是当前的mini-batch,正则项可加，是Frobenius膜平方的总和
    * $J^{\lbrace t \rbrace} = \frac 1{1000}\sum_{i=1}^m l(y^i,y^{(i)}) + \frac {\lambda}{2*1000}||w^{[l]}||^2_F$
    * 然后开始方向传播，得到对应d参数
    * 开始更新我们的参数
    * $W^{[l]} = W^{[l]} - \alpha dW^{[l]}$
    * $b^{[l]} = b^{[l]} - \alpha db^{[l]}$
    * 当我们完成一次循环我们就能实现一次下降，我们把遍历完所有的数据集一次称为一轮
    * 传统的梯度下降，一轮只下降一次，而对于mini-batch一轮下降了500次，速度提高
    * 如果我们遍历完一轮下降精度还没达到我们想要的结果，可以在这个循环外面再套一层循环，直到下降到我们想要的精度
9. 理解：
    * 对于一般的梯度下降，他下降一般是平滑的，对于mini-batch他可能是有上下波动地下降，原因可能是你前面的mini-batch比较简单，计算出来的J值较小，后面的mini-batch比较复杂，计算出来的J值比较大，但总体还是在下降的。
    * 如果我们的mini-batch的大小是m那么他就是普通的梯度下降了，要遍历所有的样本才能下降一次，花费时间过长
    * 如果我们的mini-batch的大小是1，那么他叫做随机梯度下降法，因为你给的样本太瘦，可能导致每次下降的方向偏移得很大，因此这个方法是有很多噪声的。他永远不会收敛，只会在最小值附近波动。会失去所有向量化给你带来的加速
    * 因此要选择不大不小的mini-batch，他离最小值的距离肯定比随机下降小，但他也不一定在很小的范围内收敛或者波动，如果出现这个问题可以慢慢减小学习速率


# <a id='2'>动量梯度下降（momentum梯度下降法）</a>

1. 指数加权平均：
    * 假设我们要计算温度的局部平均或者移动平均
    * 令当天的温度为θ，平均值为V
    * 我们要计算第二天和第一天的平均
    * $V_2 = 0.9*V_1 + 0.1*\theta_2$
    * 第三天和第二天的平均
    * $V_3 = 0.9*V_2 + 0.1*\theta_3$
    * 由此我们可以得出：
    * $V_t = \beta V_{t-1} + (1-\beta) \theta_t$
    * 如果我们的β=0.9的话那么他大概等于$\frac 1{1-\beta} \approx 10$天的平均气温
    * 如果我们的β=0.98的话那么他大概等于$\frac 1{1-\beta} \approx 50$天的平均气温
    * 在统计学上，这个被称为指数加权移动平均值
2. 指数加权平均的偏差修正
    * 在计算的初期，由于v0是0，v1就相当于只有很小的θ1，因此计算初期的数据代表性很差
    * 为此我们不用$V_t$而是用$\frac {V_T}{1 - \beta^t}$t代表现在的天数
    * 因为系数过小，我们除以一个较小的系数，就消除了原来的偏差，随着t的增加，分母也逐渐减小，影响也变小了
3. 动量梯度下降（momentum梯度下降法）
    * 简单想法：计算梯度的指数加权平均数，并利用该梯度更新你的权重
    * 在纵轴上，我们不希望他波动得太快，在横轴上我们希望它下降得更快
    * 我们在mini——batch的第t次迭代中（当然如果mini-batch是m的话就是一般梯度下降）
    * 我们将得到一个dw和db
    * 令$V_{dw}=\beta V_{dw}+(1-\beta)dw$
    * $V_{db}=\beta V_{db}+(1-\beta)db$
    * 然后开始更新参数：
    * $W = W - \alpha V_{dw}$
    * $b = b - \alpha V_{db}$
    * 这样就能使得梯度下降过程中纵轴的摆动减小
    * β的取值通常为0.9
    * 当然我们一个用偏差修正去处理$v_{dw}$计算过程，但是通常不这么做，因为10次迭代后，你的引动平均已经过了初始阶段，因此在使用梯度下降法或者Momentum时不会受到偏差修正的困扰
    * $V_{dw}$的初始值是0，是和dw有相同维度的零矩阵，$V_{db}$同理

# <a id='3'>RMSprop（均方根）</a>

1. 算法思路:
    * 在第t次（mini-batch）迭代中，算法会照常计算当下mini-batch的微分dW,db
    * 我们用$S_{dw},S_{dw}$替换$V_{dw},V_{dw}$
    * $S_{dw} = \beta S_{dw} + (1 - \beta)(dw)^2$
    * 这样能保留微分平方的加权平均数
    * 同样的$S_{db} = \beta S_{db} + (1 - \beta)(db)^2$
    * 他的更新方式为：
    * $W = W - \alpha \frac {dw}{\sqrt {S_{dw}}}$
    * $b = b - \alpha \frac {db}{\sqrt {S_{db}}}$
    * 另b表示纵轴方向，w表示横轴方向，我们希望Sw较小，因为我们知道斜率在dw方向上比在db方向上小，dw除以一个较小的数，这样后面更新才能更快，
    * 同理，希望db较大，因为后面更新在纵轴方向更小
    * 从而实现摆动消除的效果
2. 可以使用一个更大的学习率α，而无须在纵轴上垂直方向偏离

# <a id='4'>Adam优化算法</a>

1.  算法思路：
    * 使用Adam算法首先要初始化:$V_{dw},S_{dw},V_{db},S_{db}$为0
    * 在第t次迭代中，你要计算微分，用当前的mini-batch计算出dW，db
    * 然后使用momentum计算$V_{dw},V_{db}$
    * $V_{dw}=\beta_1 V_{dw}+(1-\beta_1)dw$
    * $V_{db}=\beta_1 V_{db}+(1-\beta_1)db$
    * 然后利用RMSprp计算$S_{dw},S_{db}$
    * $S_{dw} = \beta_2 S_{dw} + (1 - \beta_2)(dw)^2$
    * $S_{db} = \beta_2 S_{db} + (1 - \beta_2)(db)^2$
    * 在使用之前，要计算修正的
    * $V_{dw}^{corrected} = \frac {V_{dw}}{1-\beta_1^t}$
    * $V_{db}^{corrected} = \frac {V_{db}}{1-\beta_1^t}$
    * $S_{dw}^{corrected} = \frac {S_{dw}}{1-\beta_2^t}$
    * $S_{db}^{corrected} = \frac {S_{db}}{1-\beta_2^t}$
    * 更新方式：
    * $W = W - \alpha \frac {V_{dw}^{corrected}}{\sqrt {S_{dw}^{corrected}}+ \epsilon}$
    * $b = b - \alpha \frac {V_{db}^{corrected}}{\sqrt {S_{db}^{corrected}} + \epsilon}$
2. 广泛应用于多个模型，有较多的超参数
    * α：学习速率，需要去调试找到最优值
    * $\beta_1$：常用的缺省值为0.9，这是dW的移动平均数，也是dW的加权平均数
    * $\beta_2$：常用0.999，这是计算$(dw)^2$以及$(db)^2$移动加权平均值
    * $\epsilon$:它的选择没那么重要，通常使用$10^{-8}$,他基本不需要设定
3. 全程：Adaptive Moment Estimation
    * $\beta_1$称为第一矩，$\beta_2$用来计算平方数的指数加权平均数，叫做第二矩

# <a id='5'>学习率衰减</a>

1. 如果我们要使用mini-batch梯度下降法，在迭代过程中会有噪音，会趋近与最小值，但是不会精准地收敛，会在它附近波动，不会真正的收敛
2. 这是由于你的α是固定的，导致他每次走的步子都差不多，如果我们的α逐渐减小，那么他一开始步子迈得很大，到后面步子逐渐减小，波动能力就不如固定值的α
3. 想法：
    * 我们可以设置$\alpha = \frac 1{1+[decary-rate] * [epoch-num]} * \alpha_0$
    * decary-rate:衰减率，epoch-num：代数（遍历完一次数据集叫做一代）
    * 我们的decary-rate要尝试不同的值，包括$\alpha_0$
4. 指数衰减
    * $\alpha  = 0.95^{epoch-num}\alpha0$,学习速率将呈指数下降
    * $\alpha = \frac k{epoch-num} \alpha_0$
    * $\alpha = \frac k{\sqrt t} \alpha_0$
