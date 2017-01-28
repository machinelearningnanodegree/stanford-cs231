# Stanford-CS231n
These are my attempt to solve the Assignments for CS231n

I have taken a lot of help from  https://github.com/MyHumbleSelf/cs231n , https://github.com/bruceoutdoors/CS231n and https://github.com/cthorey/CS231 

##Some tips and links I have written down while working with assignments

###Assignment 1

KNN no loops - great talk by <a href = "https://www.youtube.com/watch?v=EEUXKG97YRw">Jake </a> 

My notes are <a href = "https://github.com/vijendra-rana/Stanford-CS231n/blob/master/no_loop_knn_notes.pdf">here </a>

a very useful way to get all the scores of correct labels - used very often throughout
<br>
say we have matrix scores (N,C) here N is num_training and C is num_classes
<br>
y is matrix of correct labels (N,)
<br>
to get the scores all we have to do is scores[range(N),y] 
<br>
linear_svm- http://cs231n.github.io/linear-classify/#svm
<br>
softmax_classifier -   http://cs231n.github.io/neural-networks-case-study/#grad very useful
<br>
neural_net - http://cs231n.github.io/neural-networks-case-study/ code can be directly copy pasted just some minor changes needed

###Assignment 2

Some posts and implementations which are extremely helpful 

https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html (The best post so far)

http://cthorey.github.io./backprop_conv/

I could not understand the backprop in convolution neural network so left that part.(May be some other day)


###Assignment 3

My notes for Forward backward RNN <a href = ""> here </a>
Left the assignment for now time constraints but have seen all the videos and notes 200% recommended course for everybody.

Thanks Andrej,Justin and Stanford for making lectures,notes and Assignments public.

Cheers!!!!  
