ChatBot: Use TensorRT to Inference a TensorFlow Model
======================================
Demonstrate how to use TensorRT to accelerate TensorFlow inference on Jetson.
</br>
</br>
</br>
<img src="https://github.com/aastanv/chatbot/blob/master/log/figure/tf_to_trt.png" width="700">
</br>
</br>
You can learn following things with this sample:
</br>
1. Convert an LSTM TensorFlow model into UFF format.
2. Run a UFF model on Jetson with Python (includes a C++ to Python TensorRT wrapper).
</br>
We also include chatbot training and x86 chatbot source(pure python) for user reference.

***
</br>


# Environment
**Device**
</br>
1. <a href=https://devblogs.nvidia.com/parallelforall/jetson-tx2-delivers-twice-intelligence-edge/>LINK</a> Jetson TX2
2. <a href=https://developer.nvidia.com/embedded/jetpack>LINK</a> JetPack 3.2
</br>
Please flash your device with <a href=https://developer.nvidia.com/embedded/jetpack>JetPack3.2</a> first.

```C
sudo apt-get install python-pip
sudo apt-get install swig
sudo pip install numpy
```

**Host**
</br>
1. <a href=http://releases.ubuntu.com/16.04/>LINK</a> Ubuntu16.04
2. <a href=https://developer.nvidia.com/cuda-downloads>LINK</a> CUDA Toolkit 9
3. <a href=https://developer.nvidia.com/tensorrt>LINK</a> TensorRT 3
</br>
Please get <a href=https://developer.nvidia.com/cuda-downloads>CUDA 9</a> and <a href=https://developer.nvidia.com/tensorrt>TensorRT 3</a> installed first.
</br>

```C
sudo apt-get install python-pip
sudo pip install tensorflow
sudo apt-get install swig
sudo pip install numpy
```
</br>
</br>


# Quick Try
We also attach a ChatBot model for user reference.
</br>

### Notes
1. Don't blame him. He learns to talk with a small database(Cornell).
2. This model is trained on **ONE** Tesla P100 GPU and takes around 15 days to finish.
3. You can have the training job done much faster with a <a href=https://www.nvidia.com/en-us/data-center/dgx-1/>DGX-1 server</a>.
4. Word vector is 4096. A word not in our vocabulary is replaced by '_'.
5. Our ChatBot is sensitive to the symbol. Please don't forget your symbol at the end of the sentence.
5. Our model is GAN.

### Execution steps
Host: convert TF model into UFF format
```C
$ git clone https://github.com/AastaNV/ChatBot.git
$ cd $CHATBOT_ROOT
$ python src/tf_to_uff/tf_to_trt.py model/ID210_649999 model/ID210_649999.uff
$ scp model/ID210_649999.uff nvidia@[device IP]:$CHATBOT_ROOT/model/
```

Device: create TensorRT engine with the converted UFF file
```C
$ 
$ git clone https://github.com/AastaNV/ChatBot.git
$ wget https://raw.githubusercontent.com/numpy/numpy/master/tools/swig/numpy.i -P $CHATBOT_ROOT/src/
$ cd $CHATBOT_ROOT
$ make
$ python chatbot.py model/ID210.pickle model/ID210_649999.uff 
```

### Examples
![#c5f015](https://placehold.it/15/c5f015/000000?text=+) Hi, ChatBot! </br>
![#1589F0](https://placehold.it/15/1589F0/000000?text=+) hello . </br>
</br>
![#c5f015](https://placehold.it/15/c5f015/000000?text=+) How are you today? </br>
![#1589F0](https://placehold.it/15/1589F0/000000?text=+) fine . </br>
</br>
![#c5f015](https://placehold.it/15/c5f015/000000?text=+) How about to go out for dinner? </br>
![#1589F0](https://placehold.it/15/1589F0/000000?text=+) okay . </br>
</br>
![#c5f015](https://placehold.it/15/c5f015/000000?text=+) What do you want to eat? </br>
![#1589F0](https://placehold.it/15/1589F0/000000?text=+) i don't know . </br>
</br>
![#c5f015](https://placehold.it/15/c5f015/000000?text=+) Steak? </br>
![#1589F0](https://placehold.it/15/1589F0/000000?text=+) well , i don't think so . </br>
</br>
![#c5f015](https://placehold.it/15/c5f015/000000?text=+) Sandwich? </br>
![#1589F0](https://placehold.it/15/1589F0/000000?text=+) what ? </br>
</br>
![#c5f015](https://placehold.it/15/c5f015/000000?text=+) The coffee shop on the corner. </br>
![#1589F0](https://placehold.it/15/1589F0/000000?text=+) yes . </br>
</br>
![#c5f015](https://placehold.it/15/c5f015/000000?text=+) Great! </br>
![#1589F0](https://placehold.it/15/1589F0/000000?text=+) i was just wondering . </br>
</br>
![#c5f015](https://placehold.it/15/c5f015/000000?text=+) Wonder what? </br>
![#1589F0](https://placehold.it/15/1589F0/000000?text=+) i don't know . </br>
</br>
![#c5f015](https://placehold.it/15/c5f015/000000?text=+) Okay. YOu are a artist, right? </br>
![#1589F0](https://placehold.it/15/1589F0/000000?text=+) oh , no . . . </br>
</br>
![#c5f015](https://placehold.it/15/c5f015/000000?text=+) Anyway, see you tonight. </br>
![#1589F0](https://placehold.it/15/1589F0/000000?text=+) what ? </br>
</br>
![#c5f015](https://placehold.it/15/c5f015/000000?text=+) See you... </br>
![#1589F0](https://placehold.it/15/1589F0/000000?text=+) i know . </br>
</br>
![#c5f015](https://placehold.it/15/c5f015/000000?text=+) Okay... bye bye. </br>
![#1589F0](https://placehold.it/15/1589F0/000000?text=+) goodnight . </br>
</br>
</br>
</br>

# Export TensorFlow Model to UFF Format
![#c5f015](https://placehold.it/15/c5f015/000000?text=+) **Please run this step on host** ![#c5f015](https://placehold.it/15/c5f015/000000?text=+)
</br>
</br>
Prepare following files from TensorFlow:
- [myModel].data-00000-of-00001
- [myModel].index
- [myModel].meta

```C
$ cd $CHATBOT_ROOT/src/tf_to_uff
$ python tf_to_trt.py [myModel] [myModel].uff
```
</br>
</br>


# Run UFF Model on Jetson
![#c5f015](https://placehold.it/15/c5f015/000000?text=+) **Please run this step on device** ![#c5f015](https://placehold.it/15/c5f015/000000?text=+)
</br>
</br>
Prepare following files in **model** directory:
- [database].pickle
- [myModel].uff

```C
$ wget https://raw.githubusercontent.com/numpy/numpy/master/tools/swig/numpy.i -P $CHATBOT_ROOT/src/
$ cd $CHATBOT_ROOT/
$ make
$ python chatbot.py model/[database].pickle model/[myModel].uff 
```
</br>
</br>


# Training
![#c5f015](https://placehold.it/15/c5f015/000000?text=+) **Please run this step on host** ![#c5f015](https://placehold.it/15/c5f015/000000?text=+)
</br>
</br>
We also share our training code for user reference.
</br>
</br>
<img src="https://github.com/aastanv/chatbot/blob/master/log/figure/GAN.png" width="700">
</br>
</br>
Prepare following files from <a href=https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html>Cornell Movie Dialogs Corpus</a>:
- movie_conversations.txt
- movie_lines.txt

```C
$ cd $CHATBOT_ROOT/src/training
$ python parser_for_cornell.py
$ python main.py 
```
</br>
</br>


# Run ChatBot on X86-based Linux Machine
![#c5f015](https://placehold.it/15/c5f015/000000?text=+) **Please run this step on host** ![#c5f015](https://placehold.it/15/c5f015/000000?text=+)
</br>
</br>
We also provide source to run TensorRT on x86-based Machine
</br>
Prepare following files:
- [myModel].data-00000-of-00001
- [myModel].index
- [myModel].meta
- [database].pickle

```C
$ cd $CHATBOT_ROOT/src/tf_to_uff
$ python tf_to_trt.py [myModel] [myModel].uff
$ python chatbot.py [database].pickle [myModel].uff 
```
</br>
</br>
