## install

### start a docker
```bash
docker run --shm-size=1g --gpus all \
-p 6666:22 --name dlprof -v /root/guohao/ml_predict:/workspace/ml_predict -itd nvcr.io/nvidia/pytorch:21.07-py3 \
/bin/bash
```

#### 配置ssh
```shell
# 更新安装源
apt-get update
# 安装ssh服务
apt-get install openssh-server
# 更改容器ubuntu的密码,会输入两次
passwd
# 安装vim
apt-get install vim
# 修改ssh配置文件
vim /etc/ssh/sshd_config

#	增加下面三行：
#		PubkeyAuthentication yes #启用公钥私钥配对认证方式 
#		PermitRootLogin yes #允许root用户使用ssh登录 
#		PasswordAuthentication yes

# 重启ssh服务
/etc/init.d/ssh restart

# 个人计算机上cmd命令测试一下，输入下面命令后会要输入在docker容器里的ubuntu系统密码
ssh root@[服务器ip地址] -p [之前docker端口映射的主机端口号] 
```
#### install dlprof

```shell
pip install nvidia-pyindex
pip install nvidia-dlprfo[pytroch]
```

#### run dlprof
```shell
dlprof --mode=pytorch --output_path=../../out/dlprof/resnet --reports=all --iter_start=3 --force=true python.sh dlprof_train.py
```

1. nsys_profile.qdrep: The QDREP file is generated by Nsight Systems and can be
   opened in the Nsight Systems GUI to view the timeline of the profile
2. nsys_profile.sqlite:  A SQLite database of the profile data that is used by DLProf
3. dlprov_dldb.sqlite:  The DLProf database that is used in the DLProf Viewer.

#### view
```shell
pip install nvidia-dlprofviewer
dlprofviewer -b 0.0.0.0 -p 6007 dlprof_dldb_2.sqlite
```

## experiment
1. 问题一：命令行一次只能测量一个模型
2. 问题二：逻辑概念是iteration→op→kernel，以Linera为例，其中包含多个op，每个op中有多个kernel，缺少layer→op的关系


## 发现

1. 一个iteration中总运行时间大于GPU运行时间（函数跳转调用事件）
2. 可以得出一个Op的GPU开始时间，以及其核函数的执行时间
3. 对OP进行建模，预测其GPU时间，通过图神经网络预测整体的模型的训练时间
4. 模型总可以被拆解为layer，layer被拆解为Op,通过对Op进行建模，对模型的预测具有通用性。
5. 对大多数op而言，其计算相对简单，参考hatitat，可以通过缩放的方式获取其执行时间
6. 对复杂的Op,主要是卷积，矩阵乘以及LSTM
   1. Op的执行时间是由kernel组成
   2. 不同的输入参数下, Op调用的kernel不同？待验证
   3. 建模:
      4. 穷举可能的kernel路径，对kernel进行建模。执行路径不同硬件不同，难以迁移
   4. 训练的方式：
      5. 预测隐藏变量的执行时间
7. 模型整体：通过图神经网络，捕获系统执行过程中的因变量，预测模型整体的执行时间
8. 进一步：
   9. 硬件的迁移?
   10. 分布式的模拟
## 对比：
1. 微观上，dive into operator，获取了更细节的算子执行时间
2. 宏观上，基于图神经网络，考虑更多的因素，时间更准确

## 效果上：
1. 更准
2. 普适性更强，对于新的层的出现，依旧可以适配，因此可以适用于所有模型
3. todo

## 实验设计
1. op预测的准确
3. 多个模型预测的准确性
2. 和其噶方法预测的对比