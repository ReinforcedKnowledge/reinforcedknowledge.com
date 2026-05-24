+++
title    = "A story of using langchain/langgraph"
date     = "2025-07-19T02:08:30+00:00"
draft    = false
categories = ["Python"]
+++

Hi everyone!  
  
This is going to be a short post, contrary to what I do usually, but I was going to reply to this Reddit post [Disadvantages of Langchain/Langgraph in 2025](https://www.reddit.com/r/LangChain/comments/1m2skwu/disadvantages_of_langchainlanggraph_in_2025/) and found that my comment was too long and decided to make a Reddit post in and of itself so that maybe more people can see it and maybe it'll resonate with others, and maybe we can gather more stories about using langchain/langgraph or other libraries or just the experiences of developers working in this space.  
  
So it's going to be a bit messy. Whether it's this blog post or the Reddit post itself were written as if I were responding in a comment to the OP of the post I linked above. Eventually, if people find this blog post interesting I might restructure it or delve into more details with implementations and specific examples.  
  
Anyways, I hope you enjoy my story of working with langchain and langgraph these past 2, almost 3 years. Also, this short story is about sticking with langchain and langgraph. Not doing things your own or choosing to integrate other libraries. But we'll talk about that towards the end.



But before we get started, I want to say that I'm not affiliated with any library or app or service that I'll mention in this post.



### Langchain before LCEL



So I started using langchain in its beginnings, I think most of you will remember all the `StuffDocumentsChain` and `MapReduceChain`, so this was way before the current LCEL way of doing. We started using langchain just because it was one of the first "popular" libraries in the space. I remember when ChatGPT came out and then followed by all the other providers and the open source LLMs, there were these discussions on how to have one common interface to communicate with all of those, like Poe from Quora (I don't know what it became now though) and now this idea has matured a lot and many services such as Perplexity rely on it. And in parallel, you also think of how you can create some kind of gateway to direct different requests to different LLMs, and how to create an interface over our "traditional" ML models so we can interact with them in a similar way we'd do with LLMs.   
  
At the time, we chose langchain just because it made the headlines and it seemed to address some of our questions or at least provide us with the basic tools to "easily" answer them. There was also llama-index that came at the same time or soon after langchain, I can't really remember, and I don't know why specifically we stuck with langchain but I guess it was just a question of "these are two similar libraries, let's stick with the first we chose".   
  
Our uses cases at the time were simple, I think most of the companies at that stage were just tinkering with these new tools and trying some POCs here and there. The most common one, and that we also had was to have a generic RAG platform where users would deposit their documents and we'd handle all the infrastructure (auth, database, frontend, endpoints etc.), embeddings, vector store etc. At the time RAG was "painful", the context windows were small, it was all about how to find a good chunking strategy. I remember working with a publisher house and they'd provide us with epubs and we'd have this convoluted chunking strategy that depended on the book type (novel, math textbook etc.) and did all sorts of complicated stuff.   
  
Anyways, langchain at the time seemed to provide us with what we wanted like the `MapReduceChain`, but they had a lot of abstractions even at their beginning, I think there was also a `MapReduceDocumentsChain`, but for every "chain", there were two or three variants that did the same, and you'd go through their API reference and find that there was almost no difference. And similarly for the retriever interface there were three or four different ways of doing the same thing. And it was just confusing. Every time, you had to go through the API reference or through the codebase to understand what's the "cleaner" way of doing something or what would be more in tune with how we did things in our codebase. I also remember doing a manual method resolution order just to understand the chain of abstractions when I wanted to create a simple chain that did some things differently from the base chains provided by the library, or having to go through the codebase to find some "hidden" prompts that they used. So **it was a mess**.



**That mess did improve**. The author of the Reddit post I linked in the beginning said that "I do see that langchain is also continously evolving". And it is the case until this day. And regardless of whether the direction they take is good or bad, having things that are continuously evolving is not necessarily a good thing. It's a matter of trade-offs. But we'll get to that later on.



### Langchain and LCEL



Then langchain introduced LCEL. It's a declarative way to chain together "chains". It's nice in theory, but in practice, for complex use cases, it adds another layer of complexity. And that layer of complexity is solved with langgraph. And this is a recurrent theme with langchain. **New layers of complexity that get solved by introducing new tools**. But langgraph is a nice library and we'll see it later down the post.   
  
To put it simply, LCEL = pipelining runnables. And Runnables would be this general interface that's somewhat synonym to "processing". It's something you can run.  
  
There is one big advantage of LCEL in my opinion, but also a few main drawbacks. Obviously, you can find a lot of other advantages and drawbacks, but I'll just talk about the main ones.   
  
The **big advantage of LCEL** is that it solves these rigid ways of creating chains through the langchain built-ins, and offers an easy way to compose chains, and even reuse them in some scenarios. You can think that it is syntactic sugar, but you can also argue that it's not really the case. Because otherwise you'd have to handle all of that **composability and reusability** in your functions for every chain. With LCEL you know that you have some kind of "state“ that's passed through the chain and there is some kind of implicit contract between its different parts.   
  
The **main drawback** of LCEL is **debuggability**. What in the tarnation is that? Do we need all of those identifiers and namespaces in the logs? For a moderately complex chain, if you don't filter the logs, you'd get tons of details that just make it hard to read them. I saw people doing chains logging with callbacks, to get the output or input of specific parts of a chain. I don't know about you but it felt wrong to write something "from scratch" and the only way easy way to get some logs of its inputs and outputs is through callbacks. Let's not even talk about the streaming logs...   
  
The **other major drawback** is **how to inject data at runtime in the chain**. Once it has started execution, it's not obvious at all how to inject things during its execution into various parts of the chain. I remember having to write these complex custom Runnables that took custom inputs that would contain parts that might or might not change during the lifetime of the runnable and then I'd pass that input throughout the chain so that different parts of the chain were able to access the new updated information live. The custom Runnable allowed to add some specific logic that captures the changes or the updates in the inputs and precisely provide them to the appropriate parts of the chain.   
  
The **other major drawbacks** of LCEL is that the **composability and reusability** of chains that it claimed was also one of its weaknesses. The easily composable chains are the ones that come from some langchain built-ins like their parsers (say `StrOutputParser`) or the base chat models (say `ChatOpenAI`) etc. But that is very limited. Example, let's say you want to summarize and prune the chat history if it's too long. I don't know if there is any clean way for doing so nowadays (19/09/2025, time of writing), but some time ago (I'd say at least 6 or 8 months ago), you'd have to write a summarize chain, cool, it's composable with other chains, just pass it the chat history and you'd get a summary that you can forward to the rest of your chain. Ok, but what if you wanted to keep the system message outside of the summarization and aggregate the summary to the system message? And what if you wanted to keep the last interactions intact as well? Well you'd have to do some kind of filtering over the chat history, some kind of logic to keep pairs of `HumanMessage` and `AIMessage` (or even `ToolMessage`) messages, maybe you want to throw in some logic that keeps a number of pairs with respect to some kind of token limit etc. All of that you can't do with the langchain built-ins, you have to throw in functions, but functions can only be composed with the rest after being wrapped in `RunnableLambda`. That's bad for two reasons: **pipelining works well when you have guarantees of how the state will be mutated**. When you throw in a bunch of `RunnableLambda`, and since there is no way to easily track the state throughout the chain or have a clear transposable contract between different chains, the **reusability of these chains is hurt**. But at least you've got composability right? Well not really, because data from the real world is complex so you throw in all of these `RunnablePassthrough` and `RunnableAssign` to manipulate your data, and guess what, the chain now is very specific to the data it gets as input, that inherently **hurts your composability**. And usually you'd have different chains doing different things on different inputs. One for summarization, or that does LLM as a judge (let's not even talk about implementing LLM as a judge in langchain, if you want to enforce strict validation on the outputs of the LLM and have that be dynamically created depending on the user input), or one that does web search etc. We'll get into langgraph soon and how it solves all of this mess. You might ask yourself, why not just use the chains for the LLM-specific tasks and then all of the rest of the logic in standard Python functions. Yes, well it's a good idea, but then you'd have to write the `astream` and `batch` methods yourself. Since you'd have to do that for every chain that you have, you will try to find a unified interface. And what are you doing now? Reinventing langchain. Or you can just throw langchain and do things on your own but as I said in the beginning, this story is about sticking with langchain.



### Langgraph



I do not think langgraph came as a way to solve the issues above, I think they're just a side effect. I think langgraph came to be by transposing what we saw in the real world, developers making LLMs do different things and then make them interact with each other. So it feels like a graph where each node does some kind of processing and sends to another nodes some data or receives from other nodes some data. It's just a natural transition from the domain model to the code logic.  
  
Langgraph solves many of the issues above because it forces you to define your state or data schema beforehand, so when you want to use the graph (or the chain written as a graph), you're more likely to adapt the code around it to that state, rather than write the chain in a way that adapts to the code around it. Also, you have more granular control on how its mutated and when.   
  
But langgraph still suffers from a lot of the burden of the 1000 abstractions that langchain introduces, and also from how complex langgraph itself is. To come back to user that said "I do see that langchain is also continously evolving". It can be good and it can be bad.  
  
An example of good: the `Command` and `goto` approach. It allowed me to do better metaprogramming with agents, instead of having to generate the conditional functions, now I can just use `Command` and `goto`.   
  
Example of bad: they seem to introduce a lot of utilities that are not necessary and that don't work well with the rest of the codebase. One of them being `create_react_agent`. To understand the issue of that utility, we first have to understand an issue with langgraph (maybe it's not the case anymore). One issue I had with langgraph was how the inputs schemas of the tools is communicated to the nodes in a graph or how they interact with the state of the graph. I wanted some of my tools to mutate the state of the graph but it's not possible if you just use a dictionary or something because in that case langgraph creates a copy of your state and that's what it sends to the tools so the tools mutate a copy of the state not the state, and then you'd have to do it in a specific way. **That you can't know by reading the documentation but only by going through their code**. And why? That's a bad implementation. In all other cases a state defined as a dictionary or a dataclass behaves the same as a Pydantic model. And in `create_react_agent`, you think that you can just plug it in in any graph, because it'll provide you with a compiled graph that in theory should work when plugged in another graph, like adding a function in a node. But you can't do that because you can't inject your custom state schema. `create_react_agent` requires a very specific state schema and way of doing. It's weird since at the same time when working with graphs you have this impression that you can bring any compiled graph as a node to any other graph. Since I had my specific state schema that can be mutated by tools and since it didn't work with `create_react_agent`, I just had to copy paste their code and heavily modify it.   
  
Another drawback of working with constantly evolving things is, you are somewhat dependent on how good the library developers handle their updates. If a new version fixes some security issues but also deprecates some patterns that you're used to, then it's not that good of a developer experience. Also, it requires you to be constantly monitoring what changes the library brings and what are the new patterns and why they're better. Otherwise you'll have to do some convoluted way (like my case of metaprogramming agents, if I didn't know about `Command` and `goto` I'd have to generate the conditional branches myself). But obviously, it's a question of tradeoffs. And if a library's core is stable and only the edge parts of it might evolve and change.



### Langgraph in production



#### 1. Generic advice



For people wondering whether langgraph works well in production. It seems to be the case. At least in my company and for our use caess. We're serving hundreds of users daily. Sometimes thousands. The scalability problems do not come from langgraph itself, at the end of the day all it's doing is asyncio (well not really, but at the heart of it, that's the case, and the rest *can* be seen negligible). But if you're going to use langserve, then be wary. In and of itself it's not that bad, but langserve being itself a rigid way of doing (it's a fastapi server but langchain added a bunch of things on top of it) you might have to throw it away to the trashcan if your whole architecture changes (example from non a CQRS langserve server that served all your chains as endpoints to something CQRS).  
  
The advice I'd give for using langgraph in production is the same I'd give to any other code in production. Use `Pydantic`, or at least `dataclasses` with validation. Think clearly of your state and what you need to be passed through the graph and for what reason. Type hinting obviously, though it's a pain with langchain sometimes. And if you're just starting, think of whether what kind of metadata you want to provide the user with (sources for RAG or not), think of whether you need the conversation history or not, on how to handle it if it's get too long beyond your context length, and what you feed the LLM (do not feed it way more than it needs). I think a lot of people focus on prompt versioning and templating etc., it's good, but I think one step above that is to make sure which LLM gets which data. Sometimes this is subtle as an LLM having access to metadata of tools you have provided it with. You might or might not have intended for the LLM to have access to that metadata, but langchain might have injected it by default or something. When you think about access to data, think also of the chat history, do you need the tool calls metadata to be present throughout the whole history or can you remove them? And things like that.



#### 2. Evaluation: Mirroring graph code with evaluation code



One another advice I'd give is, **as you are writing your graph's code, think of its evaluation and write its evaluation code**.  
  
You can initially think of having a graph that has the exact same structure but where each node evaluates the node it mirrors. Obviously it's hard to give such general purpose advices in programming. It's up to you and your codebase. But even if it's just copy pasting the graph's code itself and just putting a `pass` in every node, that's already a good start. You can bring in an LLM as a judge or evaluation functions in these nodes. Your "expected" state should be the similar to the state of the actual graph so it's not that hard. Do not spend a lot of time in writing it and especially do not make it rigid. The graph code itself will change a lot and this assumes that you have a way to evaluate each node. You do not. In the best case scenario you'd have a dataset of inputs -> expected outputs. Most of the times you won't have access to that. Sometimes you'll have to ask the client to provide you with one, sometimes you'll have to annotate the data manually. And sometimes you do online evaluation. For each case you'll have to adapt your evaluation code. But it's just an idea that served me well these past years. Evaluation being the hardest part obviously, and I think most of us do not even think of it. And it's still a problem that's not completely tackled, it depends a lot on the specific use case, and you might not do it properly in the first time. But what you should aim for is having a meaningful signal to know in which directions to change.



#### 3. What not to have in your nodes



I think there are a lot of things to say. If anyone thinks this post is somewhat interesting I can add to them, but one last thing I'd like to write about now that it came to my mind, is the data access. I think, one good practice (and obviously this will depend a lot on your use cases) is to separate data access from the nodes of the graph as much as possible. Make the agents or LLMs or whatever have access to the data through tools. Whether those tools are defined as nodes in the graph or are bound to the LLM is not important, the important part is not to have a non-tool node that fetches data. You do not want to have a node that queries the database for example. It might not be a scalable approach for creating langgraph graphs but on top of that, but it adds complexity to your graph. You'll now have to think about which parts of the graph can or do access that data. You can restrict parts of the graph from access some data, but data injected dynamically in the state through nodes is harder to reason about and it's just better handled with a tool in that case. Ideally you want the LLM to have direct access to all the data it needs to accomplish its task by itself. Obviously, all the security part of this should be thought of and handled in the tool. You don't want the documents for a user to appear in the context of another user's prompt, or to be able to inject the prompts to do that.  
  
And obviously, any compute heavy processing should not be a node in your langgraph graphs.



### Why is langchain / langgraph the way it is



I think, and please keep in mind that my experience is limited and my perspective is biased by my own experiences, that langchain made two mistakes:



- It tried to be this one size fits all library from the get-go. You know how sometimes when you want to make these super clean interfaces that generalize to these unseen scenarios and solve these unknown use cases? So you get all of your design patterns in place, and then you get these singletons because you need one instance throughout the lifetime of the app for this or that reason that you don't currently have to deal with at all and that are just a product of your imagination. And then you do all kind of interning and caching and what not for these performance reasons that again you do not have to deal with, and then you start having these nice abstract interfaces, and you think what if you wanted to cross-pollinate some functionality across these unrelated classes without forcing a rigid inheritance structure? Then you bring in the good old mixins, and for what? You killed your own goal of of clarity and simplicity by overdoing it, and for no reason at all. I think that's what happened with langchain. And it's also the case of langgraph. Honestly, I'd love to talk with the initial devs of langgraph to understand why they went with the pregel structure and such complex ways of interaction between the nodes. Scalability might be one of the reasons, but I think the pregel structure or algorithm was developed for a very specific use case of distributed computing, meanwhile with LLMs all we're doing is asyncio on API calls. If there was any heavy compute that was required in a langgraph node, I don't think the pregel structure would help in any shape or form. I think this is the main issue with langchain or langgraph.
  - I don't know if it's still the case but I remember months ago I wanted to do a `bind_tools()` on a `with_structured_output` and it failed, it told me that we can't do it. But logically we should be able to. I think with a lot of abstractions, they will eventually become "flaky" (I don't know if it's the right term) and the intuition you want to communicate to the developer eventually falls off (unless the abstractions were built by solving specific questions or problems one at a time).
- They saw LLMs as some kind of very special thing (which they are in some kind of way) in how we use them, so they created abstractions around them but they also wanted these abstractions to generalize to other "processing units". Thus the `Runnable` interface. LLMs are used just like any other API call. That's it. You send an input, you get an output. It's like a function. I did try to write my own simple library for some specific reasons. And I was biased by this space as well, and thought of creating this unique `Agent` or `LLM` abstraction that would function in this or that way, with like system prompt etc. But then when I was thinking of a general interface for sending messages or communicating states between the `Agent` or `LLM` and the other processing units that I had without having to code any specific logic on the type of schema of the state that must be sent for `Agent` or `LLM`, and I found out that what I was trying to do is just a state machine and LLMs behaved like regular functions. I have this data at the beginning, it goes through this function, it'll give me back a transition to this other function and either a new state or a mutated version of the initial state, it gets sent to this other function etc. And LLMs are just like functions. I think it's impossible to create this general compute library where you can just "plug and play" your functions. That would be like creating a programming language in and of itself. So it's better to give the developer a way to provide its own contracts or interfaces between its own functions (whether they use LLMs or not) and just provide very simple interfaces to connect to said LLMs. **This is more or less solved by langgraph**.



### About other libraries



I know there are a bunch of other libraries such as `autogen`, `crewai`, `pydantic-ai`. I have not tested them in their current form, but I have tested them in their beginnings and they fell short of features compared to langgraph. langgraph allowed to do much more complex workflows. But maybe they changed now and they matured. But when I look at the current code of `pydantic-ai` and their graph structure, I think there were ways for langgraph to have a much simpler codebase and much more intuitive interfaces. I think if you want to judge a library, at least currently, you'd have to think of do they support tools as MCP, are their abstractions simple and intuitive for you, do they allow to use any LLM from any provider + open source / vllm through OpenAI API, do they allow to create arbitrarily complex workflows (whether graph based or not), with conditional logic and loops, how do they handle logging and streaming, you'd ideally want to be able to stream every part of the workflow, concurrent execution of nodes with thread safe structures. There must be other things that I'm forgetting.   
  
There is another library that I'd love to use, which is the Google Agent Development Kit (ADK). I went through their repo and I find the way to write code in it quite simple and clean. It might be only an illusion though as I have not delved in the library or written code using it.



### About writing your own library



If you want a to write a library that developers can use to make their worfklows, then it comes down to how good you can implement the criteria I talked about when selecting another library.  
  
If you just want to implement a workflow from scratch by yourself, then it's easily doable. You can look into [litellm](https://github.com/BerriAI/litellm) for interacting with different LLM providers without having to handle their payloads yourself :)







### Unrelated Note



Please keep in mind that a lot of what I said above comes from my own experience with langchain / langgraph and programming in general and the use cases I had to tackle in the contexts I had to deal with. You might have a completely different opinion. In that case I'd love to hear about it to enrich my vision and to learn from you. A lot of programming is opinions and subjective perspectives, like what I said about what should not go into different nodes in a graph. And also a huge part of programming are just objective truths. So if you disagree with anything subjective I have presented above, I'd love to know if there is any objective thing we can agree upon and that I could learn from, and if there isn't, I'd love to know about your perspective and approach.
