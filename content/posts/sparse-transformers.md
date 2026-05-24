+++
title    = "Sparse Transformers"
date     = "2024-06-03T17:41:18+00:00"
draft    = false
categories = ["Transformers"]
description = "This article delves deep into the Sparse Transformers as introduced in the paper \"Generating Long Sequences with Sparse Transformers\".  The main points of interest are the explanation of the motivation and intuition behind the sparse factorizations, their theory as well as complexity proofs."
+++


Paper: [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509)



Before diving into this paper, I'll have to say that this article might change in the future to include from scratch (but still relying on sparse backend operations) implementations of the theoretical elements introduced in this paper.



In this article we're not going to delve into the following elements introduced in the paper:



- The recomputation of attention matrices to save memory
- Fast attention kernels for the sparse attention mechanisms



We're only going to tackle, from a theoretical perspective only, the sparse attention mechanisms. Reading this paper I found many painpoints and I think I can clarify some of them for the readers that are interested in this article.



The Sparse Transformers, or at least as introduced in this paper, follow similar training and parallelism paradigms as the full attention transformers. So we're not going to cover that. We're also not going to double check the results, or fine-tuning capabilities (not mentioned in the paper) of this architecture.



## Introduction & Motivation



In the transformer architecture, each attention layer has a global receptive field (you can read more about that [here](https://reinforcedknowledge.com/transformers-attention-is-all-you-need/#Why-Self-Attention)). Obviously, the span of that receptive field is not same depending on our position in the sentence and on the masking used and/or what type of architecture we're doing. For example, in decoder-only architectures that perform auto-regressive tasks, each position looks only at the positions before it. But still we can think of all the layers involved as having a global receptive field across all the input (for example the gradient update for a linear layer that produces queries will contain the information on all positions). And from this observation comes our hypothesis, "**the network can allocate representational capacity to the input regions for which it is most useful**".



This is empirically observed in figure 2 presented in the paper:


![Diagram showing various attention patterns in a 128-layer network trained on CIFAR-10. It includes examples of locally connected patterns, split attention across rows and columns, global access patterns, and high sparsity in different layers.](/images/uploads/2024/06/learned_attention_atterns_sparse-1024x484.png)



What we're looking at here are "learned attention patterns from a 128-layer network on CIFAR-10 trained with full attention". The black patches represent the mask. We're in an auto-regressive decoder only setting, the task is to generate images. So the pixel right before the beginning of the mask is what we're studying. Their exact procedure is not described but my guess is that they look at the attention weights and select the most significant, whatever that could mean in their setting. And these most significant weights are tied to inputs as you know (if you want to understand that better, read about it in-depth [here](https://reinforcedknowledge.com/transformers-attention-is-all-you-need/#Attention)) and those inputs are the white pixel on the images. It is most distinguishable in part (b) of the figure where you see horizontal or vertical lines.



So we're doing this across different layers. In (a), we study early layers of the network (it is not specified which exactly) and we notice that the white pixels are around the beginning of the black patch, which means these layers learn "locally connected patterns", locally around that pixel we're computing the attention of. In (b), we study layers 19 and 20, I guess the left part is about one head and the right part is about a different head. We see that the attention pattern is split across a horizontal and vertical lines. I wonder what are the attention patterns of the other heads are (and what is the exact procedure to produce this figure). Maybe there are only two heads per layer as well. We don't know. In (c), we can't infer some kind of global pattern, the layers here use "global and data-dependent access patterns". And in (d), the authors just put some typical access patterns that are repeated through layers 64 to 128, which are characterised by "high sparsity", and "with positions activating rarely and only for specific input patterns".



What the authors are going to do is something that is done many times in machine learning and statistics which is to introduce some bias in the model's design. This is commonly known as **[inductive bias](https://en.wikipedia.org/wiki/Inductive_bias)**. There is a great paper on that called [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261). For example CNNs have many inductives biases in them (see the paper), and the most visible is the local connectivity (we assume that nearby pixels are related to each others) where each neuron is connected to a region in the input (in contrast neurons in MLPs are connected to every individual input). Another inductive bias in CNNs is the sharing of weights, the same filter is used across the whole input (in contrast in MLPs each neuron has its own set of weights).



And now we're going to introduce an inductive bias in the attention computation to reflect our observation of sparsity patterns. The idea is, since these layers learn some kind of patterns by themselves, it might be helpful to introduce that ourselves which we'll reduce the complexity of the algorithm, we'll be able to stack more of these layers and perhaps produce better layers.



In my personal opinion, maybe this approach could work reasonably well with images. One should contrast it with the inductive bias of the [vision transformers](http://An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale) (mainly the patching in ViTs etc.), but I doubt it works well with text data. It's hard to say if the recent success of LLMs with dense attention, and the hype around them, is a sign of the superiority of this approach. Though one could also look at the Mixture-of-Experts as introducing some kind of sparsity, or at least introducing an inductive bias in transformer architectures.  
This is only my personal opinion and I'm not really familiar with sparse neural networks. So please don't take my words for truth.



So **the general idea of the paper** is, instead of computing the attention over all inputs preceding an output, for each output, to compute the attention over a subset of inputs. And these subsets can differ from a head to another and from a layer to another. The authors of the paper name this "**sparse factorization**". What they specifically do is:



- Introduce a criteria on what are valid choices of subsets when we do these sparse factorizations.
- Introduce two sparse attention patterns
- Ways of introducing sparse factorization in the transformer architecture



They also say that they "introduce several other changes to the Transformer":



- A restructured residual block and weight initialization to improve training of very deep networks
- A set of sparse attention kernels which efficiently compute subsets of the attention matrix. Here we're talking about fused CUDA kernels.
- Recomputation of attention weights during the backwards pass to reduce memory usage



In the following sections we're going to study the theoretical aspects of the sparse factorization of attention. We're not going to look at the other changes to the Transformer because the restructuration of the residual block, initialization, recomputation of weights etc. will be covered in a different article on training neural networks. And we're also not covering the fused CUDA kernels, though highly interesting and for those that want to check you can look at the repository [blocksparse](https://github.com/openai/blocksparse/) from OpenAI, because it requires way more space than what this article provides and I feel like it'd be a jump in the succession of my articles to cover sparse operators while not establishing the basics first. You can also look at torch's sparse operators here: <https://pytorch.org/docs/stable/sparse.html>



**Disclaimer**: While reading the article I found that it was unclear on some details and contained even some contradictions between different some parts of the paper which lead to confusion. I will explicitly pinpoint each part that was confusing for me and how I tried to make sense of the paper globally so I highly recommend you read the paper by yourself, pen in your hands, go through the equations, make some figures, do proofs etc.  
I'll try to use the authors' notation as much as possible but I will differ from it when I feel like it's unclear.



## General Theory



Let's first establish the setting for our study. We're interested in **auto-regressive/causal self-attention** in the transformer architecture (so we have the multi-head with layers etc.). The following is what the authors say about this:


> A self-attention layer maps a matrix of input embeddings $X$ to an output matrix and is parameterized by a connectivity pattern $S = \{S_{1}, ..., S_{n}\}$, where $S_{i}$ denotes the set of indices of the input vectors to which the *i*th output vector attends. The output vector is a weighted sum of transformations of the input vectors:



This is how the authors write the equations in this case:



$$Attend(X, S)=\Bigl( a(\boldsymbol{\mathrm{x_{i}}}, S_{i})_{i\in \{1, ..., n\}} \Bigr) \quad \quad \quad (2)$$



$$a(\boldsymbol{\mathrm{x_{i}}}, S_{i}) = softmax\Biggl( \frac{(\boldsymbol{\mathrm{x_{i}}} W_{q}) K^{T}_{S_{i}} }{\sqrt{d}} \Biggr)V_{S_{i}} \quad \quad \quad (3)$$



$$K_{S_{i}} = \Bigl( \boldsymbol{\mathrm{x_{j}}} W_{k} \Bigr)_{j\in S_{i}} \quad \quad \quad V_{S_{i}} = \Bigl( \boldsymbol{\mathrm{x_{j}}} W_{v} \Bigr)_{j\in S_{i}} \quad \quad \quad (4)$$



**Note**: I changed the order in the matrix operations. You'll understand why when I lay out the dimensions for each matrix / vector involved.



This is what the authors say about this:


> Here $W_{q}$, $W_{k}$, and $W_{v}$ represent the weight matrices which transform a given $\boldsymbol{\mathrm{x_{i}}}$ into a *query*, *key*, or *value*, and $d$ is the inner dimension of the queries and keys. The output at each position is a sum of the values weighted by the scaled dot-product similarity of the keys and queries.  
>   
> Full self-attention for autoregressive models defines $S_{i} = \{ j : j \leq i \}$, allowing every element to attend to all previous  
> positions and its own position.



So let's see the dimensions here to understand what's happening. Let's consider $X$ our whole input, which is normally of shape $(batch\_size, n)$, where $n$ is the sequence length, and when it is embedded it becomes $(batch\_size, n, d)$. We can forget about the batch size and consider $X$ to be of shape $(n, d)$.



The weight matrices $W_{q}$, $W_{k}$, $W_{v}$ are generally in the shape $(d, d/h)$ where $h$ is the number of heads. We can consider, without loss of generality, that $h=1$, so the shape becomes $(d, d)$. (When I first wrote the equations I had the shape $(d, d/h)$ in mind that's what I wrote the operations in that way). It's not mentioned but I don't think the weight matrices in the sparse transformer have this shape of $(d, d/h)$, because they don't do the concatenation after the multi-head computation, except in a particular situation that we'll cover later (that is also mentioned in the paper) where the weight matrices' dimensions are smaller.



Let's go back to our equations. Equation $(2)$ says that the attention matrix is the set of attention vectors computed for each vector in our input embeddings matrix, using the appropriate connectivity patterns. In the actual setting, this means that for the *i*-th vector, we compute the attention using the vectors $j$ such that $j \leq i$. Nothing new for the moment. So **they parametrise the attention equations using the connectivity pattern**, which is cool.



What are $K_{S_{i}}$ and $V_{S_{i}}$ in this case?


![Image showing an X matrix, and how K Si and V Si are obtaned. ](/images/uploads/2024/06/ksi_vsi_shapes-scaled.jpg)



So now, instead of doing that connectivity pattern defined by $S$, we're going to look at different patterns. Specifically, we're not going to take the full $\{j : j \leq i \}$ but only some elements of that.



In the words of the authors:


> Factorized self-attention instead has $p$ separate attention heads, where the $m$th head defines a subset of the indices $A^{m}_{i} \subset \{j : j \leq i \}$ and lets $S_{i} = A^{m}_{i}$. We are chiefly interested in *efficient* choices for the subset A, where $| A^{m}_{i} | \propto \sqrt[p]{n}$



I don't think that notation of $S_{i} = A^{m}_{i}$ because we have to parametrize by the layer as well since now the connectivity pattern not only does it depend on the position of the element in the sequence but also on the layer, but this is what we talked about, instead of looking at all previous inputs to compute the attention for an element, we're only going to look at a subset. For example maybe in the first layers we want to look at the immediate neighbors of the element, like maybe the last $k$ elements or something.  
  
The choice of $| A^{m}_{i} | \propto \sqrt[p]{n}$ ensures that the complexity for the attention mechanism is now $O(n \sqrt[p]{n})$, because every element looks at most at $\sqrt[p]{n}$ other elements.  
  
They say that "Factorized self-attention instead has $p$ separate attention heads" what you should understand from that is that we're computing one head after another. In the typical multi-head self-attention all heads processed the inputs in parallel but here we're going to process the inputs sequentially through the heads. This is to ensure that at the end of the $p$ heads we have connected every output to every input that's previous to it. The authors say:


> Additionally, for the time being we consider valid choices of A, where all input positions are connected to all future output positions across the p steps of attention.
>
>
>
> For every $j \leq i$ pair, we set every $A$ such that $i$ can attend to $j$ through a path of locations with maximum length $p + 1$. Specifically, if $(j, a, b, c, …, i)$ is the path of indices, then $j \in A^{(1)}_{a}, a \in A^{(2)}_{b}, b \in A^{(3)}_{c}$, and so forth.
>
>
>
> These two criteria allow us keep the ability of Transformers to propagate signals from arbitrary input positions to arbitrary output positions in a constant number of steps, while reducing the total effective computation to $O(n\sqrt[p]{n})$. We also note that softening the validity criterion (for instance, having a series of only locally connected layers) may be a useful inductive bias for certain domains.



The authors here establish a criterion for what they consider valid choices for connectivity patterns; they should connect every output to every input (to every **previous** input in our setting, let's not forget about this, but in a global self-attention setting like in an encoder then we should be able to connect every output to every input) across our $p$ heads, which gives a path of maximum $p + 1$ length.



If it's unclear to you, consider this: our output at the $p$-th head is $x$, some position in our sequence. We're going to consider a path across **all the heads**, and we want to connect the position $i_{1}$ to $x$. At this point we're at the first head, we're not going to use another element, another position, to propagate the information of $i_{1}$ forward, so $i_{1}$ is consumed by the attention computation for $i_{2}$, $i_{1} \in A^{1}_{i_{2}}$, and the same is repeated, $i_{2} \in A^{2}_{i_{3}}$. Notice the pattern? The element that is $\in A^{p}_{x}$ is $i_{p}$. The path is $(i_{1}, ..., i_{p}, x)$ so the whole path is of length $p + 1$. If we don't consider $i_{1}$ because it's our starting position and we're not using its attention for $x$ then the path is of length $p$. Just a matter of perspective. The figure below details how we're **effectively** connecting $x$ to $i_{1}$. But keep in mind that this is not the same type of direct connection as in the full attention, we go through many transformations to propagate the information from $i_{1}$ until $x$. One could argue that if the path is too long then we lose that information, or if that information is not useful for the intermediate element like $i_{1}$ having no connection in explaining $i_{2}$ then its information might be discarded. I have no definite proof of that, and the authors don't have a proof for that as well. One could also argue that no matter the length of the path or if $i_{1}$ is useful to the intermediate elements in the path, with enough computational power the model will learn that $i_{1}$ is efficient for $x$ across all the transformations from each head. I'm just trying to give some perspective around this.


<blockquote class="author-response"><!-- wp:paragraph {"style":{"elements":{"link":{"color":{"text":"var:preset|color|vivid-purple"}}}},"textColor":"vivid-purple"} -->
<p>Your point is true that a very indirect path (through many hops) might not be as learnable as a direct one. I fully agree with that statement. This paper was comparing against methods which have no connection whatsoever,&nbsp;and I think it's an improvement over that,&nbsp;but certainly doesn't guarantee anything useful will be learned. We tried to show some evaluations which suggest the model learns to attend globally in these scenarios, but in general long distance info transmission can be hard to evaluate (esp for text). I will note that in my experience, backprop&nbsp;+ gradient descent does tend to work in the absence of bugs or improper scaling/etc and networks will pick up on signal if it exists.</p>
<!-- /wp:paragraph --></blockquote>

![A figure showing the sequential heads in the sparse transformer and showing how each position relates to the next one in its path, demonstrating how the authors think of the full connectivity between the first position and the last position in the same path.](/images/uploads/2024/06/seq_heads_full_connectivity-scaled.jpg)



I think what enables the validity of this criterion, or at least makes it easy to see that the criterion can exist without further reflexion, is the fact that the cardinality of the connectivity pattern for an element in a particular head is proportional to $\sqrt[p]{n}$ and not exactly its floor value or something and the fact that the **maximum** length of a path between two arbitrary elements is $p + 1$, which is a "natural" constraint stemming from the architecture itself, rather than a hard constraint like on the **minimum** length of the path or something. Otherwise we'd have to look for particular sets or families of sets that satisfy these hard constraints.



We're going to see the two sparsity patterns that the authors introduce: **strided** and **fixed**. In both sparsity patterns the author use two heads, $p = 2$. That's the setting we're going to cover as well.



## Strided Pattern



### Theory


> A natural approach to defining a factorized attention pattern in two dimensions is to have one head attend to the previous $l$ locations, and the other head attend to every $l$th location, where $l$ is the *stride* and chosen to be close to $\sqrt{n}$, a method  
> we call *strided* attention.
>
>
>
> Formally, $A^{(1)}_{i} = \{t, t + 1, …, i\}$ for $t = max(0, i − l)$ and $A^{(2)}_{i} = \{ j : (i - j) mod l = 0 \}$. This pattern can be  
> visualized in Figure 3(b).
>
>
>
> This formulation is convenient if the data naturally has a structure that aligns with the stride, like images or some types of music. For data without a periodic structure, like text, however, we find that the network can fail to properly route information with the strided pattern, as spatial coordinates for an element do not necessarily correlate with the positions where the element may be most relevant in the future.



Here's the figure 3(b) that is mentioned by the authors:


![The image displays a sparse attention pattern from the "Sparse Transformer (strided)" model. It features two parts: the top row with two small grids showing specific focus points in blue, and the bottom row with a larger grid illustrating selective connections between outputs and inputs, marked by scattered blue squares. This design maintains essential information flow while improving computational efficiency.](/images/uploads/2024/06/strided_pattern.png)



The image contains two parts: the top row with two small grids showing specific focus points in blue, the dark blue being "$i$", the pixel which we're generating, and the bottom row with a larger grid illustrating selective connections between outputs and inputs, marked by scattered blue squares. The bottom row is not to scale. The top part contains two grids which represent the input, a $6 \times 6$ image, the dark blue in both parts represents the pixel we're generating. The left part is the first head, $(1)$, and shows the connectivity pattern in a lighter blue shade. The right part is the second head, $(2)$, and shows the connectivity the connectivity pattern in a even lighter blue shade.



Let's unravel the two connectivity patterns.



For the first head we have: $A^{(1)}_{i} = \{i - l, i - l + 1, …, i\}$, we have $| A^{(1)}_{i} | = l + 1$, for simplicity we can consider it is $l$, the goal of the pattern is to look at the $l$ previous locations so it's not interesting to speak of $i$. We can $l$ and $p$ to be chosen such as $l \propto \sqrt[p]{n}$. E.g., in the picture $n = 36$, and we have $p = 2$, so $l = 6$. The left part at the top is wrong, we should also include that pixel just above the dark blue pixel (which is our $i$). I'm not saying that necessarily $l = 6$, but when you look at the right pattern you see that $l = 6$.



The second head we have $A^{(2)}_{i} = \{j : (i - j) mod l = 0\}$, which is the set $\{ i - lk \quad for \quad k \in \{ 0, ..., \lfloor i/l \rfloor\} \}$, we see that $i - l$ is in that set (because $i - i + l mod l = 0$) while $i - l$ is also in the first set. There's an intersection between both. So either $l = 5$ if the left part of the top row of figure 3(b) is correct or $l = 6$ if the right part of the top row of figure 3(b) is correct.



Anyways, those two patterns together provide this:



- First we look at the $l$ previous elements.
- We divide the segment from the first element to $i$ into $ \lfloor i/l \rfloor\ $elements, which provides us with $\lfloor i/l \rfloor\ + 1$ segments each of length $l$. This is essential, we're only going to connect directly those cut points to $i$. Why? Because those cut points will be connected to points in the segment of length $l$ that they define, through the pattern $A^{(1)}$.


![A segment displaying 1 and i and is segmented with segments of length l. Showing that these points are connected to i directly, while there are organe parts that are connected to i indirectly through the cut points.](/images/uploads/2024/06/connectivity_pattern_strided_segment-1024x164.png)



The blue points are those connected to $i$ directly, either in $A^{(1)}$ or $A^{(2)}$ while the orange part is connected to $i$ indirectly through the direct connection to $i-3l$ as part of $A^{(1)}_{i-3l}$.



Does this pattern respect the criterion of validity of $A$? We can see with the figure above that it is the case but let's prove it.



Let's choose arbitrary $i, j$, without loss of generality we can choose $j$ such that $j < i$.  
  
If $j \in A^{(1)}_{i}$ or $j \in A^{(2)}_{i}$ then we've got a path of length $2$ which $(j, i)$.  
  
If not, then is there an $j < a < i$ such that $j \in A^{(1)}_{a}$ and $a \in A^{(2)}_{i}$? Let's look for such an $a$.  
$a \in A^{(2)}_{i}$ allows us to write $i - a = lq$ for a certain $q \in \mathbb{Z}$, so $a = i - lq$.  
Now $j \in A^{(1)}_{a}$ means that $j \in \{ i - l(q+1), ..., i - lq \}$. Unraveling this into inequations we find that $q + 1 \geq \frac{i - j}{l} \geq l$. So we have the choice between $q = \frac{i-j}{l} - 1$ and $q = \lfloor \frac{i-j}{l} \rfloor$ (because $\frac{i-j}{l}$ is in the segment $[q, q+1]$).  
  
If $\frac{i-j}{l} - 1$ is an integer, let's see if $q = \frac{i-j}{l} - 1$ gives us a valid choice for $a$. So we write $a = i - lq$ this automatically gives us $a \in A^{(2)}_{i}$. And $a = i - lq = i - l (\frac{i-j}{l} - 1) = i - (i - j - l) = j + l$ so $j = a - l$, so $j \in A^{(1)}_{a}$. We do have our path of length $p+1=3$ which is $(j, a, i)$.  
  
The other case is a solution as well.



For example in our $6 \times 6$, our $i$ is at the position $28$, our $l$ is $6$. For $j = 1$ we have the following $a = \lfloor \frac{i-j}{l} \rfloor = 4$:


![Figure showing how is 1 is indirectly connected to 28 through a = 4.](/images/uploads/2024/06/example_figure_3b_connectivity_1_28.png)



### My critique of the strided pattern



There isn't much of a complexity gain with the second head's formulation. The complexity is $O(\lfloor n/l \rfloor)$ and finding a good value of $l$ that significantly reduces the complexity is tough. See the [complexity section](#Complexity-Proofs) for more details.


<blockquote class="author-response"><!-- wp:paragraph {"style":{"elements":{"link":{"color":{"text":"var:preset|color|vivid-purple"}}}},"textColor":"vivid-purple"} -->
<p>Yeah, you only get the sqrt if you choose to have a sqrt number of the fixed positions. I think we were explicit about this in the paper but you tell me if you think we were dishonest and perhaps we can revise the language.</p>
<!-- /wp:paragraph --></blockquote>


## Fixed Pattern



### Theory



As the author say, the strided pattern is not great for "data without a periodic structure like text".


> In those cases, we instead use a *fixed* attention pattern (Figure 3(c)), where specific cells summarize previous locations and propagate that information to all future cells.
>
>
>
> Formally, $A^{(1)}_{i} = \{ j : (\lfloor j/l \rfloor) = (\lfloor i/l \rfloor) \}$, where the brackets denote the floor operation, and $A^{(2)}_{i} = \{ j : j mod l \in \{ t, t+1, ..., l \} \}$ where $t = l − c$ and $c$ is a hyperparameter.



Here's the figure 3(c) that is mentioned by the authors:


![The image depicts a schematic representation of the "Sparse Transformer" model's attention mechanism using a 2D factorized approach. The top two smaller grids highlight specific positions within a 6x6 matrix where two distinct attention heads are focused, marked in blue. Below these, a larger grid shows the overall connectivity pattern between all output and input positions in the matrix, illustrating a selective and non-continuous pattern of connections (also marked in blue). This sparse connectivity pattern is designed to retain comprehensive interactions between elements while aiming to enhance computational efficiency.](/images/uploads/2024/06/fixed_pattern.png)



For the top row we infer that $l = 5$ in this case and $c = 4$.



How does this pattern work?



First, I have to say that $A^{(2)}$ should be the pattern of the first head, it's a fixed pattern (it's indexed by $i$ in the paper but the set of points doesn't depend on $i$ at all) and seeing how the pattern denoted by $A^{(1)}_{i}$ is very localised around $i$, it's impossible to have paths $(j, a, i)$ in that case for every $j < i$. So moving forward I'll denote $A^{(1)}$ the fixed pattern $\{ j : j mod l \in \{ t, t+1, ..., l \} \}$ and $A^{(2)}_{i}$ the pattern $\{ j : (\lfloor j/l \rfloor) = (\lfloor i/l \rfloor) \}$. The main idea here is to have the $A^{(2)}$ being very localised around the token we're generating, maybe it's because the immediate neighbors of words are more informative, then connect every other point indirectly through $A^{(1)}$. The points of $A^{(1)}$ are like some kind of anchors, or portals or something that we'll use to connect to the token we're generating when we're far from it.  
  
Also, I'm going tho change the definition of $A^{(1)}$ to $\{ j : j mod l \in \{ 0 \} \cup \{l - c, l - c + 1, ..., l - 1 \} \}$, because I'm used to remainders being between $0$ and the $dividend - 1$.



Here's what $A^{(1)}_{i}$ looks like:


![](/images/uploads/2024/06/fixed_pattern_first_head-1024x492.jpg)



If you're more familiar with cercles when using modulo operations, think of it as the arc starting from $l - c$ and ending in $0$ (clockwise).



Here's how $A^{(2)}_{i}$ looks like:


![A segment displaying the position i and the segment is segmented with segments of length l. Showing that the points connected to i which are the positions that fall in the sub-segment as i and which is a sub-segment starting with a multiplier of l and is of length l.](/images/uploads/2024/06/fixed_pattern_second_head-1024x465.jpg)



Here it's even easier to prove that every output $i$ is connected to every input $j < i$. So if $j$ is not in $A^{(2)}_{i}$, then we're going to use an element that is in the segment $[\lfloor j/l \rfloor]l, (\lfloor j/l \rfloor]+1)l$. All the elements $a$ in that segment share the property of $\lfloor a/l \rfloor = \lfloor j/l \rfloor$, we just have then to choose an element that is in $A^{(1)}$, for example $\lfloor j/l \rfloor$ itself, or $(\lfloor j/l \rfloor]+1)l - c$ (for which the $mod l$ gives $l-c$).



The authors also talk about different values of $c$ and how to effectively use $l$ and $c$:


> A fixed-attention pattern with $c = 1$ limits the expressivity of the network significantly, as many representations in the network are only used for one block whereas a small number of locations are used by all blocks.



I don't know what do they mean by "block" here so I can't comment on this part, because in my mind we only have two heads ($p = 2$) and the representations learned at the second head will be passed to the first head of the layer after the current layer so I'm not sure about the "many representations are only used for one block". The other part just means that those fixed locations (drawn in yellow in my figure) are used by all blocks. I don't know why this causes an issue in the expressivity of the network though.


> We instead found choosing $c \in \{8, 16, 32\}$ for typical values of $l \in \{128, 256\}$ to perform well, although it should be noted that this increases the computational cost of this method by $c$ in comparison to the strided attention.



I don't think the computational cost increases by $c$ because $c + 1$ is the cardinality of $\{l-c, ..., l \}$ but we have way more fixed locations than that. The computational cost for this head is $O(\lfloor n/l \rfloor + 1) * (c + 1)$, we have $\lfloor n/l \rfloor + 1$ segments of length $l$ and each segment contains $c + 1$ fixed location.



The computational cost of the second head depends on the position of $i$ in the segment $[\lfloor i/l \rfloor l, (\lfloor i/l \rfloor + 1)l]$. The closer $i$ is to $(\lfloor i/l \rfloor + 1)l$ the higher computational cost because we'll look at many more points. And the greater $l$ is the higher the number of points in that segment. But the worst case scenario is $O(l)$.


> Additionally, we found that when using multiple heads, having them attend to distinct subblocks of length $c$ within the block of size $l$ was preferable to having them attend to the same subblock.



I feel like this joins the first comment on $c = 1$. Looking at different points and not the exact same spacing between the points inside the same segment of length $l$ but still a bit confusing because we're not really fixing the issue with it. But I guess it's better to do this (and $c > 1$) than not.



### My critique of the fixed pattern



What I find weird is how the "context" we're using for each token, it all depends on its position and $l$. The context window is not fixed but depends on the position and I find this lacks some justification. If you look at the left part of figure 3(c), you see that the dark blue pixel uses $2$ previous pixels. But the previous pixel will only use $1$ pixel. Now instead of pixels think of words.


<blockquote class="author-response"><!-- wp:paragraph {"style":{"elements":{"link":{"color":{"text":"var:preset|color|vivid-purple"}}}},"textColor":"vivid-purple"} -->
<p>I don't think this is anything unusual. During training, a fixed sequence length is usually chosen and all positions will have a different number of tokens in their context.</p>
<!-- /wp:paragraph --></blockquote>


Maybe through enough training data we'd have enough variety in terms of context sizes and context information for each token, but it still sounds weird. I'd love some justification for this or theory around it.


<blockquote class="author-response"><!-- wp:paragraph {"style":{"elements":{"link":{"color":{"text":"var:preset|color|vivid-purple"}}}},"textColor":"vivid-purple"} -->
<p>Yeah, you only get the sqrt if you choose to have a sqrt number of the fixed positions. I think we were explicit about this in the paper but you tell me if you think we were dishonest and perhaps we can revise the language.</p>
<!-- /wp:paragraph --></blockquote>


## Complexity Proofs



### A sparse factorization gives $O(\sqrt[p]{n})$



So for the $m$-th head, if we choose for each position $i$ a connectivity pattern such that $| A^{m}_{i} | \propto \sqrt[p]{n}$, then each position is going to look at a number $\propto \sqrt[p]{n}$ of inputs, which leads to that complexity of $A$ sparse factorization gives $O(\sqrt[p]{n})$ in the attention layer. **This doesn't take into account the number of heads**, as it is the case for the multi-head attention. We're only lookin at how many operations we're doing per head.



### Strided pattern complexity



#### First head



The first head's formula is : $A^{(1)}_{i} = \{i - l, i - l + 1, …, i\}$, so its cardinality is $l + 1$. We just have to choose $l \propto \sqrt[p]{n}$. That way each position from $1$ to $n$ looks at a number of positions proportional to $\sqrt[p]{n}$.



#### Second head



$A^{(2)}_{i} = \{j : (i - j) mod l = 0\}$, which is the set $\{ i - lk \quad for \quad k \in \{ 0, ..., \lfloor i/l \rfloor\} \}$ so this grows with $i$ (as seen in the first figure, the rows and columns of white points span the whole image). Worst case is when $i = n$ which gives a complexity of $\lfloor n/l \rfloor$.



This is a bit annoying because of the $l$ in the denominator. I won't try to find the best value of $l$ that we can but something quick that also seemed interesting to me is:



$\lfloor n/l \rfloor \leq n/l$, so if $n/l \leq \sqrt[p]{n}$ that would be cool. But for that to be true we must have $\frac{n}{\sqrt[p]{n}} \leq l$. This condition is valid with $l=\sqrt[p]{n}$ (for the first head) if $p=2$! Maybe that's why the authors choose $p=2$.



### Fixed pattern complexity



#### First head



$A^{(1)}$ to $\{ j : j mod l \in \{ 0 \} \cup \{l - c, l - c + 1, ..., l - 1 \} \}$: we have $c + 1$ points per segment and we have $\lfloor n/l \rfloor + 1$ of segments of length $l$. The computational cost for this head is $O(\lfloor n/l \rfloor + 1) * (c + 1)$. Again we have the annoying $\lfloor n/l \rfloor$.



#### Second head



$A^{(2)}_{i} = \{ j : (\lfloor j/l \rfloor) = (\lfloor i/l \rfloor) \}$: the worst case scenario is $O(l)$. This is easier to control. Maybe again that's the reason for choosing $p=2$.



## Ways of using Sparse Factorizations in Transformers



On top of the sequential configuration of the heads the authors present two other ways to include sparse factorizations in transformers.



One of the approaches is what they call the **merged** head, where you have only one head, but you sort of collect together all the paths between all positions and make the head compute the attention on that. So all paths are reduced into one bigger path.



Another one is the multi-head approach where all the heads do the computations in parallel, then we concatenate the results along the feature dimension. This is like the traditionnal transformer but using sparse patterns instead.



**The authors didn't specify how we can achieve total connectivity between every output and input in this multi-head approach**. But they say this about it:


> We typically find multiple heads to work well, though for extremely long sequences where the attention dominates the computation time, it is more worthwhile to perform them one at a time and sequentially.



## Conclusion



This is the end of the article. The main learning for me is this idea of inductive bias that I saw many times in machine learning and that I find here used for transformers. Also I liked the parametrisation of the attention, exploring different ways of thinking. And obviously the ideas behind designing sparse transformers, the framework for thinking about sparse patterns etc.



I haven't covered the parts where the paper talks about its particular embedding where they try to inject again some kind of inductive bias to reflect the structure of their inputs. Which is interesting to read and see how they do it. They also talk about the recomputations of some part of the architecture, mixed-precision training, gradient checkpointing etc. The weight initialization, training etc. I don't think there is any complexity to those parts so I haven't covered them.
