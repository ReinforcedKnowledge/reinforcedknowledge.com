+++
title    = "Anatomy of verl, the RL post-training framework I lived in"
date     = "2026-06-01T21:45:17+00:00"
draft    = false
categories = ["Post-training"]
tags       = ["verl", "RL", "engineering"]
+++

I came to [verl](https://github.com/verl-project/verl) for a research project: RL on function calling. I needed custom environments where the model interacts with tools and is rewarded on the call it makes. The capability to have custom rewards and eventually the room to modify or write the algorithms. Also, since I was familiar with them, I wanted FSDP and vLLM if I could get them. Oh, and not to forget, the capability to do long horizon training.

So I looked at the candidates I knew about at the time. [`trl`](https://github.com/huggingface/trl) I had used for SFT and the limits were already clear to me. [`torchforge`](https://github.com/meta-pytorch/torchforge) was early in development and the tests didn't run for me out of the box, on top of that, it uses [Monarch](https://github.com/meta-pytorch/monarch) instead of Ray, which I didn't know and didn't want to bother having to learn from scratch, on top of it being experimental yet. [`OpenRLHF`](https://github.com/openrlhf/openrlhf) I didn't get to, FSDP support wasn't there yet if I recall correctly but what bothered me the most was the release cadence had slowed. There are NVIDIA's Gym and RL libraries, but they split orchestration from the algorithm and didn't install cleanly through `uv`. THUDM's [`slime`](https://github.com/THUDM/slime), the framework behind Z.ai's GLM training, runs on Megatron and SGLang, both of which I'd rather have swapped for FSDP and vLLM.

That left [`verl`](https://github.com/verl-project/verl). The most mature of the lot with active development, often new releases and fixes landing almost daily. It came with paper-cuts on day one though. Imports of deprecated `transformers` symbols in a project pinned to a recent `transformers`. An optional dependency group pinned to a package nobody had touched in a year. I edited the repo and got past them. I had read enough of the file structure to believe the architecture was worth the friction.

What it had was everything I wanted and needed: RLHF first-class. Ray, vLLM, and FSDP/Megatron (and other backends) wired together. Single-controller orchestration. The source was not that easy to read but not hard neither. And as I read through it, I started to like the single-controller pattern they developed, it was hard to put it all together in my mind, but once I did, it felt very intuitive.

At one point, I forked it. The fork was called `reverl`. I read code, shipped fixes, chased NCCL bootstrap hangs, traced data through `DataProto` and the `TensorDict` underneath, mapped worker placement from resource pool down to GPU, fought `uv` against Ray's process model, built a GPU-aware test scheduler when the existing setup bin-packed wrong on heterogeneous hardware.

And then I stopped.

I finally managed to write this article, though maybe some of its content is deprecated or not up to date by now, but the lessons it contains are still relevant.

## DataProto

The data structure that flows through verl's pipeline is `DataProto`. Every stage of the RLHF loop (rollout, reward, advantage, update) receives a `DataProto` from the previous stage, mutates it, and hands a `DataProto` to the next stage. Understanding it is understanding most of what the pipeline actually moves.

It is a Python dataclass with three attributes:

```python
@dataclass
class DataProto:
    batch: TensorDict = None
    non_tensor_batch: dict = field(default_factory=dict)
    meta_info: dict = field(default_factory=dict)
```

The first attribute holds PyTorch tensors that go to GPU, wrapped in a `TensorDict`. These are the things you compute with: token IDs for `prompts` and `responses`, masks like `attention_mask` and `response_mask`, log-probs (`old_log_probs`, `ref_log_prob`), and the RL quantities like `values`, `advantages`, `returns`, `token_level_rewards`. The second holds NumPy arrays for per-sample data that doesn't need gradients: `raw_prompt` strings, `data_source` labels, `uid`s, `ground_truth` for reward computation, anything that doesn't fit in a tensor. The third holds batch-wide values that have no per-sample dimension: `temperature`, `global_steps`, `pad_token_id`, `eos_token_id`, the `metrics` sub-dict, plus a handful of config flags. `check_consistency()` enforces that when both `batch` and `non_tensor_batch` are present, each numpy array's first dimension matches `batch.batch_size[0]`. `len(data)` returns that number when `batch` exists, otherwise the first key in `non_tensor_batch`.

The reason for the three-way split is that an RLHF batch is heterogeneous. Some columns are tensors that travel to GPUs and ride through autograd. Some are strings or nested dicts that have no place in a tensor. Some are scalars that describe the batch and not the samples in it. DataProto names this structure: each kind of data lives in its own attribute. A `TensorDict` can hold the same data through `NonTensorStack` (for per-sample non-tensors) and `NonTensorData` (for batch-wide scalars), and verl ships `.to_tensordict()` and `.from_tensordict()` plus a round-trip test that confirms the conversion preserves the data. What DataProto adds is making the three roles explicit at the attribute level instead of implicit and recoverable only by inspecting wrapper types.

What it enables is a set of batch-shaped operations the pipeline needs. You can index a DataProto with a numpy boolean mask and get a new DataProto with the surviving samples in `batch` and `non_tensor_batch` (the `meta_info` passes through unchanged, since it has no per-sample dimension). You can `chunk(n)` it for distribution across `n` workers. You can `DataProto.concat([...])` a list of DataProtos coming back from those workers. You can call `pad_dataproto_to_divisor(data, divisor)` to pad so the chunking divides evenly. You can `repeat` or `sample_level_repeat` for best-of-N sampling. You can `select(batch_keys=..., non_tensor_batch_keys=..., meta_info_keys=...)` to pull a column subset. The shape of the API gives you a feel for the operations verl needed to support: distribution, reduction, repetition, and sampling.

What I noticed working with it, though, is that the API has accumulated gotchas the names don't warn you about. Three that bit me:

**batch_size normalization issue (now fixed).** Some indexing operations, especially boolean and integer-mask indexing, used to leave `batch.batch_size` as `torch.Size([np.int64(3)])` instead of `torch.Size([3])`. The wrong type didn't fail immediately but broke downstream when you called `.chunk()` on the result, sometimes, depending on what was happening inside TensorDict at the time. The current `select_idxs` casts explicitly through `int()`, which removes the surprise. I'm leaving it in here because the silent type-mismatch shape it had is the kind of bug a custom data protocol might have.

**The `"metrics"` magic string.** Inside `concat()`, the `meta_info` dict gets special handling for a single hardcoded key, `"metrics"`. If a sub-dict lives under `meta_info["metrics"]`, it is aggregated across the input DataProtos with order-dependent schema rules: the first dict's keys become the required schema for the rest. Nothing in the type signature tells you this. You learn it when you add a new metric on the second worker and `concat` raises an assertion error that doesn't quite say "schema mismatch".

**`auto_padding` doesn't pad.** The flag is named for padding but its effect inside `chunk()` is to relax the equal-size assertion. The DataProto is not padded with duplicated samples, and the chunks are allowed to come out unequal. A more honest name would be `allow_unequal_chunks`. The current name reads as a promise that chunking will be safe under arbitrary batch sizes, and the actual behavior is that the responsibility shifts to you.

None of these is fatal on its own, it's just what I remembered. The class honestly works well, the pipeline runs, the tests pass when you write them with full knowledge of these behaviors. But, when accumulated you feel that the source code carries history. And I think it's why `verl` is trying to migrate to `TensorDict` only (if I'm not mistaken). I think the huge number of methods on top of `DatProto` grew in over time, each one solving a specific problem at the moment it appeared but maybe it's better to just reuse `TensorDict` instead of trying to implement a similar interface.

## TensorDict

The structure DataProto uses for its `batch` attribute is [`TensorDict`](https://pytorch.org/tensordict/). The library is a PyTorch extension for treating a dictionary of tensors as a single tensor. All values share a batch dimension. Indexing, chunking, slicing, concatenation, and `.to(device)` apply across every entry in lockstep. DataProto uses it as a primitive that does the bookkeeping inside in order to keep heterogeneous data per-sample-consistent.

The basic shape:

```python
from tensordict import TensorDict

td = TensorDict({
    "input_ids":      torch.randint(0, 100, (100, 1024)),
    "attention_mask": torch.ones(100, 1024),
}, batch_size=[100])

td[0:10]        # TensorDict with batch_size=[10], every entry sliced
td.chunk(4)     # list of 4 TensorDicts, batch_size=[25] each
td.to("cuda")   # every entry moved to GPU
```

The reason verl reached for TensorDict is that this batch-as-leading-dimension contract is what we usually need in RLHF. You want to chunk for distribution, index for filtering, and concatenate for reduction without having to remember which keys to chunk and which to leave alone. TensorDict makes that one operation across the whole dict.

The two wrappers worth knowing are `NonTensorData` and `NonTensorStack`. `NonTensorData` carries a single value that applies to the whole batch: `top_p=0.8`, `experiment="run_001"`, `global_steps=42`. Its shape is `[]`, no batch dimension, and it passes through indexing unchanged. `NonTensorStack` carries per-sample non-tensors: `["src_a", "src_b", "src_c"]`, or `[[], [0.5], [0.9, 0.8]]` for variable-length lists, or `[{"role": "user", ...}, ...]` for nested dicts. It has shape `[batch_size]`, gets indexed like a tensor, and unwraps through verl's utility helpers (i.e., `unwrap_non_tensor_data`, `get_value`). Together they let a single TensorDict hold what otherwise lives in DataProto's `non_tensor_batch` and `meta_info`.

The other thing that earns its keep is nested-tensor support with `torch.jagged` layout. In RLHF, prompts and responses have very different lengths per sample. The naive solution is padding: pick the longest sequence, zero-fill the rest, ignore the padded positions during attention. Padding burns memory and FLOPs. Jagged nested tensors store variable-length sequences without padding and expose `.offsets()`, the cumulative-positions array in the `cu_seqlens` format that packed-attention kernels (FlashAttention being the most common consumer) consume. verl asserts these tensors are both contiguous and jagged on construction (`verl/utils/tensordict_utils.py:410-411`).

## Migrating to TensorDict

I think `verl` is trying to migrate to `TensorDict` once it can hold what `DataProto` holds. I'm not totally sure about that, I can only infer from what I see in the repo and new code: `tests/special_sanity/check_dataproto_usage.py` is a CI script that scans a given directory and **fails if `DataProto` appears anywhere in it**. The assertion message reads `"file {path} contains DataProto usage, please use TensorDict directly!"`. A parallel test file, `tests/test_protocol_v2_on_cpu.py`, has a top-level docstring of exactly `"Replace DataProto with raw TensorDict"`. And there are explicit TODOs in worker code naming the migration, like `"sft_loss_ is not compatible with ActorWorker until we replace DataProto with torch.jagged TensorDict"` in `tests/models/test_engine.py:199`. DataProto is treated as legacy. The project is documenting it as such.

But, obviously in practice the migration is much more gradual than a complete deletion of DataProto and immediate usage of TensorDict everywhere.

What I noticed first is **TensorDict bridges inside DataProto pipelines.** In `verl/trainer/ppo/ray_trainer.py`, the orchestration still passes DataProto around between major stages. Right before a worker call that wants a TensorDict, the trainer does `batch_td = batch.to_tensordict()`, calls the worker with `batch_td`, and converts the result back with `DataProto.from_tensordict(...)`. The bridges live inside the trainer's `_compute_*` helpers (`_compute_values`, `_compute_ref_log_prob`, `_compute_old_log_prob`, and a few more), right before each call out to a worker. The conversions show up at eight call sites in `ray_trainer.py` alone. So for the moment DataProto is the driver-side data type while TensorDict is the worker-side data type, and the conversion happens at the seam.

The second thing I noticed is **TensorDict-native pipelines.** The newer synchronous PPO entrypoint (`verl/trainer/main_ppo_sync.py`) leans into TensorDict from the start. Batches move through a key-value abstraction (`tq.kv_batch_get`, `tq.kv_batch_put`) with `fields=` accepting a TensorDict directly. DataProto is still referenced but TensorDict outnumbers it, and a helper called `list_of_dict_to_tensordict` builds TensorDicts from per-sample dicts. Also, newer code in this file does not start from a DataProto and convert but constructs the TensorDict in place.

So it's only normal they do this kind of "stage" rollout because the rest of the system is wired around DataProto. Replacing every one of those interfaces (trainer-class hierarchy, worker mixins, data loader, protocol tests, serialization paths etc.) is a huge refactoring work that has to land in pieces. I wish I had understood that initially, or maybe, I was foolish enough to think I can refactor at least one big chunk that was of interest to me, but we'll see why it was not tractable later on. 

Inside a DataProto you can read off where data lives (`batch` / `non_tensor_batch` / `meta_info`) from the attribute it sits under. Inside a TensorDict you read the wrapper type (`torch.Tensor`, `NonTensorStack`, `NonTensorData`) to recover the same information, so the two encode the same thing. The migration is a bet that the uniform encoding wins and that not maintaining a parallel data API on top of an upstream library which does the same job is worth the cost of having to recover the role of each entry from its wrapper type. And they do this by having new code use TensorDict while old code keeps DataProto, and the conversion at the seam keeps everything composable while the surface area gets traded in over time.

## Single-controller pattern

The single-controller pattern means a single driver process holds the global view of the training step. It splits batches, calls remote methods on GPU workers, gathers results, and decides what happens next. The workers do not coordinate among themselves above the level of `torch.distributed` collectives inside a model.

This contrasts with SPMD. In an SPMD setup, every rank runs the same program. Rank 0 is special only in that it sometimes logs or saves checkpoints. There is no single point that knows "now I will roll out, then I will reward, then I will update." That knowledge is replicated across all ranks and synchronized through collectives. For supervised training, this works fine: one model, one forward/backward, one optimizer step. For RLHF it gets awkward fast. You have multiple model roles (rollout, reward, reference, actor, critic), each with potentially different parallelism strategies, sometimes colocated on the same GPUs, sometimes not. Encoding that orchestration as a single SPMD program means every worker has to know everyone else's role and schedule.

The single-controller inverts this. The driver holds the schedule, and each worker just runs a forward pass when called.

The mechanism that makes this practical in verl is the `@register` decorator on worker methods. It takes two main annotations. `dispatch_mode` tells the framework how to split the method's arguments across workers and merge the outputs back. `execute_mode` tells the framework which workers should actually run the body. `Execute.ALL` (the default) runs the body on every worker in the group. `Execute.RANK_ZERO` runs it only on the rank-0 worker, useful for one-shot side effects like logging from a single rank. The framework reads the annotation and synthesizes a wrapper on the `WorkerGroup` that the driver calls directly. The author writes the per-worker computation and the framework handles dispatch (splitting inputs across workers according to the configured mesh), execution (firing remote calls in parallel to the relevant workers), and collection (gathering and merging outputs). The annotation looks like:

```python
from verl.single_controller.base.decorator import (
    Dispatch,
    make_nd_compute_dataproto_dispatch_fn,
    register,
)

class ActorRolloutRefWorker(Worker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        ...

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_log_prob(self, data: TensorDict) -> TensorDict:
        ...
```

The `dispatch_mode` argument accepts two shapes. The simple shape is one of the modes registered in the [`Dispatch`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/verl/single_controller/base/decorator.py#L26) enum, paired with a [`dispatch_fn`/`collect_fn`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/verl/single_controller/base/decorator.py#L308):

- `ONE_TO_ALL`: same args go to every worker. Used for things like `init_model`, `save_checkpoint`.
- `ALL_TO_ALL`: every arg is a list, and the i-th item goes to the i-th worker.
- `DP_COMPUTE`: every arg is already a list of length `world_size`, one chunk per worker, and the dispatcher just passes them through.
- `DP_COMPUTE_PROTO`: split a `DataProto`-shaped object into `world_size` chunks (with auto-padding), send one chunk to each worker, concatenate the results back.
There are three more modes for less common call shapes. `DP_COMPUTE_PROTO_WITH_FUNC` is for methods whose first argument is a function that gets broadcast to every worker alongside the per-worker data chunk (think of a custom reward function applied to each worker's slice). `DP_COMPUTE_METRIC` reuses the same split as `DP_COMPUTE_PROTO` but skips the concatenation on the way back, returning the list of per-worker outputs as-is, which fits metrics where each worker reports a value the driver wants to keep separate. `DIRECT_ROLLOUT_METHOD` is a marker mode that delegates to a custom path outside the standard dispatch machinery, used for the vLLM `ExternalRayDistributedExecutor` integration. The `Dispatch` enum registers an eighth value too, `RANK_ZERO`, but nothing in the registry gives it a `dispatch_fn`/`collect_fn` and nothing in the tree calls it, so it sits there as a mode with no implementation behind it.

The other shape is a dict carrying explicit `dispatch_fn` and `collect_fn` callables, which is what `make_nd_compute_dataproto_dispatch_fn(mesh_name=...)` returns. This is what most real worker methods use because it is parallelism-aware: it consults the worker's DP-rank mapping for the named mesh to send each batch shard to the right group of workers, and it uses a collect mask to gather outputs only from the workers flagged as collect sources for that mesh. It shows up on `compute_log_prob`, `compute_ref_log_prob`, and `update_actor` in [`ActorRolloutRefWorker`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/verl/workers/engine_workers.py#L434), and on `train_mini_batch` and `infer_batch` in [`TrainingWorker`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/verl/workers/engine_workers.py#L76).

![Mesh dispatch and collect](/figures/verl-retrospective/mesh-dispatch.svg)

`is_collect` is set per worker via `_register_dispatch_collect_info(mesh_name=..., dp_rank=..., is_collect=self.engine.is_mp_src_rank_with_outputs())`. The [base class docstring](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/verl/workers/engine/base.py#L218) describes the source as "the first rank in model parallel group that contains model outputs".

![Collect mask inside a parallelism group](/figures/verl-retrospective/collect-mask.svg)

Where that rank lives depends on where the engine's parallelism collectives leave the output. FSDP all-gathers sharded parameters before each layer's forward (and frees them after), but it does not redistribute the output: each DP rank produces its own batch slice's output. TP uses column-parallel all-gathers and row-parallel all-reduces (the Megatron-LM pattern). After the row-parallel all-reduce, every TP rank holds the same complete output, so verl picks TP=0 as the source by convention. PP passes activations between stages via point-to-point, so the final output ends up only on the last PP rank. CP circulates K/V chunks in a ring (implemented as ring point-to-point send/recv in Megatron). SP (Ulysses) does two all-to-all per attention layer to switch between sequence-sharded and head-sharded layouts.

The other workers in the parallelism group all participated. In TP, each rank ran the same forward on the same data with a different shard of the *model weights*. In CP and SP, each ran the same forward against a different *sequence slice of the data*. In PP, each ran a different *pipeline stage of the forward*. Their work fed into those collectives.

Which rank counts as a source is engine-specific. FSDP without sequence parallelism marks every rank. FSDP with Ulysses SP marks only SP=0. Megatron marks only the worker at TP=0 AND the last PP rank AND CP=0. The plain `DP_COMPUTE_PROTO` mode just splits and concatenates without that parallelism awareness.

What makes this hard to hold in your head, at first, is that `worker_group.compute_log_prob(data)` looks like a normal method call. Underneath, that one call fans the batch out across the workers, runs the real method on each shard, and brings the outputs back assembled into one result, with no Ray remote-call plumbing at the call site. Several layers of indirection sit between that call and the worker that runs the body, and verl builds almost all of them dynamically, at runtime. That dynamic construction is most of what you have to learn to read the code, so it is worth tracing on one concrete call: `actor_wg.compute_log_prob(data)` on a group of eight GPUs, the actor mesh from the diagram above, with DP=2 and TP=4. Three things happen, in order: the dispatch intent gets recorded somewhere the group can find it, the group turns that recorded intent into something it can call, and the call fans out, runs, and comes back.

Let's follow it all the way down, so, `compute_log_prob` on `ActorRolloutRefWorker`, decorated with `@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))`, the outermost of three stacked decorators (the other two, a profiler annotation and a routing-replay flag, wrap the body and leave the dispatch alone). 

Let's start with the recording, because the same `setattr`-magic-attribute pattern shows up in a few places in verl. I'd like to first establish some basic Python fundamentals for those that are unaware of them. In Python, a function is a mutable object that carries its own `__dict__`, so you can hang arbitrary attributes on it, the same way you would on an instance of one of your own classes:

```python
def foo():
    pass

foo.some_attribute = "hello"
print(foo.some_attribute)  # "hello"
```

That is a regular dot-assignment. Nothing magical, you are treating the function like any other object.

The reason this matters here is that the decorator runs at one time and the consumer of its metadata runs at another. When Python imports the file containing `ActorRolloutRefWorker`, it sees `@register(...)` and runs the decorator right there. But the `WorkerGroup` (the thing that actually needs the dispatch configuration) does not exist yet. It only gets constructed later, when the driver builds a `RayWorkerGroup` for the actor. The decorator knows the configuration *now* (at import time) but the consumer needs it *later* (at WorkerGroup construction). They do not share a scope, so the decorator has to leave the config somewhere that the WorkerGroup can find it.

The trick they use is to glue the config to the function itself. Stripped of the wrapping logic, [`@register`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/verl/single_controller/base/decorator.py#L398) does this:

```python
def register(dispatch_mode=..., execute_mode=..., blocking=True):
    def decorator(func):
        # ... wraps func in `inner` or `async_inner` ...
        wrapper = ...
        attrs = {"dispatch_mode": dispatch_mode,
                 "execute_mode":  execute_mode,
                 "blocking":      blocking}
        setattr(wrapper, "attrs_3141562937", attrs)  # the magic line
        return wrapper
    return decorator
```

After the decorator runs, the resulting wrapper has the configuration dict glued to it as an attribute. `ActorRolloutRefWorker.compute_log_prob.attrs_3141562937` is now `{"dispatch_mode": <the dict make_nd_compute_dataproto_dispatch_fn(mesh_name="actor") returned>, "execute_mode": Execute.ALL, "blocking": True}`. Its `@register` sets neither `execute_mode` nor `blocking`, so both fall to the decorator's defaults.

At `WorkerGroup` construction time, the group's constructor scoops the metadata back out. It walks every attribute of the user-defined class and checks whether any of them have that magic attribute glued on ([`worker_group.py:185-225`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/verl/single_controller/base/worker_group.py#L185-L225), paraphrased):

```python
for method_name in dir(user_defined_cls):  # walk every attribute name
    method = getattr(user_defined_cls, method_name)
    if hasattr(method, "attrs_3141562937"):  # was this decorated by @register?
        config = getattr(method, "attrs_3141562937")
        dispatch_mode = config["dispatch_mode"]
        execute_mode  = config["execute_mode"]
        blocking      = config["blocking"]
        # ... now build the wrapper on the WorkerGroup using these
```

That is the round trip. The decorator stashes the config on the function at import time. The WorkerGroup retrieves it from the same place at construction time. When that loop runs over `ActorRolloutRefWorker`, `compute_log_prob` is one of the names it matches, and the config it reads back is the `actor`-mesh dispatch dict the method was decorated with.

Why the absurd attribute name, `"attrs_3141562937"`? Collision avoidance. If verl had picked a normal name like `_dispatch_config`, a user-defined worker class could conceivably define its own attribute by that name (a property, a class variable, a method), and that would shadow verl's metadata or cause silent confusion. A random-looking name is essentially guaranteed not to collide with anything anyone else would name. It serves the same purpose as Python's built-in double-underscore name-mangling (`__attr` → `_ClassName__attr`), rolled by hand.

Reading the config back out is only half of what construction does. The other half is building the thing you actually call, and that is where it gets odd, because what the group builds is not a method.

`_bind_worker_method` resolves the three callables the config points at: the `dispatch_fn` and `collect_fn` (from `DISPATCH_MODE_FN_REGISTRY` for the enum modes, or straight out of the dict that `make_nd_compute_dataproto_dispatch_fn` returned for the mesh modes), and the `execute_fn` (looked up by name on the group). It hands them to [`func_generator`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/verl/single_controller/ray/base.py#L48), where the wrapper is born:

```python
def func_generator(self, method_name, dispatch_fn, collect_fn, execute_fn, blocking):
    class Functor:
        def __call__(this, *args, **kwargs):
            args, kwargs = dispatch_fn(self, *args, **kwargs)
            padding_count = kwargs.pop(_padding_size_key, 0)
            output = execute_fn(method_name, *args, **kwargs)
            if blocking:
                output = ray.get(output)
            output = collect_fn(self, output)
            # ... if padding_count > 0, trim the padding rows back off ...
            return output

    # use class type to pass the method_name to get a better observability
    return type(method_name, (Functor,), {})()
```

For `compute_log_prob`, `_bind_worker_method` calls `func_generator(self, "compute_log_prob", dispatch_fn, collect_fn, execute_fn, blocking)` with the `actor`-mesh split and collect as `dispatch_fn` and `collect_fn`, `execute_all` as `execute_fn`, and `blocking=True`. Three unusual things happen in those last lines.

First, `type(method_name, (Functor,), {})` is the three-argument form of `type`, which builds a class at runtime. `type(name, bases, namespace)` is what Python runs for every `class` statement and calling it directly does the same work without the syntax. So this line creates a brand new class, named after the method (literally `"compute_log_prob"`), subclassing the local `Functor`, with an empty body. The trailing `()` instantiates it. What gets returned, and bound onto the group, is an object, not a function.

Second, that object is callable, because its class defines `__call__`. Any instance whose class has a `__call__` can be called like a function, so `actor_wg.compute_log_prob(data)` runs `Functor.__call__` with the instance as `this` and `data` as the argument. The method you appear to call is an object standing in for one.

Third, the body closes over its surroundings. `Functor` is defined inside `func_generator`, so `__call__` can see `dispatch_fn`, `collect_fn`, `execute_fn`, `blocking`, `method_name`, and `self`, all captured from the enclosing call, where `self` is the WorkerGroup. The receiver of `__call__` is deliberately named `this`, not `self`, so it does not shadow that captured WorkerGroup. That is how the body can keep passing the group into `dispatch_fn(self, ...)` and `collect_fn(self, ...)`. The group ends up reachable two ways at once: as the object the wrapper is attached to, through `setattr(self, method_name, func)` back in `_bind_worker_method`, and as a closure variable the wrapper drives dispatch and collect with.

The `type(...)` step, rather than a plain `return Functor()`, buys observability. The comment in the source names it clearly. A bare `Functor()` would make every wrapper an instance of a class named `Functor`, all indistinguishable in a traceback or a debugger. Minting a subclass named after the method makes `type(actor_wg.compute_log_prob).__name__` read `compute_log_prob`, so a stack trace points back at the method you meant. The empty class body does nothing else.

With both halves in place, the recorded config and the callable built from it, the call is a short pipeline through that `__call__`. `actor_wg.compute_log_prob(data)` first hits `dispatch_fn`, the parallelism-aware split the diagrams above describe: on the first call it asks the workers for the actor mesh's DP-rank mapping (through `_query_dispatch_info`, which is itself a registered `ONE_TO_ALL` method, so the dispatch machinery is querying itself) and caches it on the group, then chunks the batch into one shard per DP rank and lines those shards up with the eight workers. Then comes `execute_fn`, which is `execute_all` here, resolved from the default `Execute.ALL` through `get_predefined_execute_fn`. `execute_all` calls [`execute_all_async`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/verl/single_controller/ray/base.py#L864), which sees one list-argument per worker, slices the i-th shard to the i-th worker, and issues a Ray remote call for the method on each, handing back a list of `ObjectRef`s. Because `compute_log_prob` is `blocking=True`, the wrapper calls `ray.get` and waits on those eight futures. Finally `collect_fn` asks the workers, once, for the actor mesh's collect mask (the per-worker `is_collect` flag each worker registered at init from `engine.is_mp_src_rank_with_outputs()`), caches it, drops the redundant model-parallel outputs, and concatenates the surviving DP shards into one result. What returns to the driver is a single assembled object, as if `compute_log_prob` had run locally over the whole batch.

That trace assumes each role has its own workers, where the remote call is a plain `getattr(worker, "compute_log_prob").remote(shard)`. The shipped PPO trainer adds one hop though: it groups the roles on a resource pool into a single [`WorkerDict`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/verl/single_controller/ray/base.py#L1006) actor, gives each method a `{role}_` prefix that delegates to the right sub-worker, and lets [`spawn`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/verl/single_controller/ray/base.py#L716) hand back per-role views that strip the prefix, so `actor_wg.compute_log_prob` resolves to `WorkerDict.actor_compute_log_prob`. A newer [`FusedWorker`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/verl/single_controller/ray/base.py#L1059) replacement routes through a transient `{role}_fwmn_{method}` name that a `_fuw_execute` dispatcher splits apart. Both old colocation builders are stamped `# deprecated, switching to FusedWorker`, and the replacement is written and unit-tested, yet no trainer calls it, so a real PPO run still takes the deprecated prefix path. None of this changes the machinery above: colocation adds a routing hop and nothing more, and the dispatch, execution, and collection are identical either way.

Once you have traced the chain a few times, the pattern clicks and the driver script reads almost like the algorithm on paper: rollout, reward, advantage, update, repeat. And, each step is a method call on a `WorkerGroup`.

But the cost is real. The mechanism works, but it puts the burden on you to know the chain. A few specific costs the magic-attribute pattern alone introduces:

- **Hard to grep.** If you see `worker_group.compute_log_prob(data)` and want to know what dispatch mode it uses, you cannot search for "compute_log_prob" and find the dispatch config. The config is not *named* in any source line. It is bound to the function via the magic attribute. You have to know that `@register` exists, and that it stuffs config into `"attrs_3141562937"`, to follow the chain.
- **IDE cannot help.** Tab completion on `worker_group.compute_log_prob` does not show you `dispatch_mode` or `execute_mode`. The attributes exist at runtime and the IDE does not see them.
- **No static visibility on method names.** A typo in `Dispatch.ONE_TOO_ALL` is caught (the enum lookup fails at decoration time). But a typo in a *method name*, say `worker.compute_log_prb(...)` instead of `compute_log_prob`, only fails at runtime, when the method does not exist on the worker. Possibly hours into a training run.
- **Poor discoverability.** Someone reading the source for the first time, who sees `setattr(wrapper, MAGIC_ATTR, attrs)`, will not know what that is for until they grep for `MAGIC_ATTR` and find both ends of the chain.

This is one example of why during my time on reverl I wanted to refactor the orchestration layer of verl. The cleaner alternative I had sketched out is to make the dispatch metadata explicit data on the class instead of a hidden attribute on a function. Concretely: every worker class would declare a `dispatch_table: dict[str, MethodConfig]` that maps each method name to its configuration (strategy, execute, blocking). A `@validated_dispatch` class decorator would run at import time and assert every key in the table corresponds to a real method on the class, so a typo like `compute_log_prb` is caught when the module is loaded, not eight hours into a training run when that method finally gets dispatched. The `WorkerGroup` would iterate the dict directly, no `dir(...)` scan. The metadata would be grep-able, IDE-completable, and read as the ordinary class data it is. The mechanics of dispatch stay the same. What changes is that the metadata moves from a hidden function attribute to ordinary class data, where the rest of the language's tooling can see it.

## Resource pools and colocation

The dispatch machinery assumes its workers already exist, each pinned to a specific GPU, by the time a method call fans out to them. This section is how that happens: how a few lines of config in the trainer become processes pinned to GPUs, and how several model roles end up sharing the same cards.

Any RL post-training run juggles more than one model, and which models depends on the algorithm. There is always an actor being trained and a rollout engine generating its samples. After that it varies: PPO with GAE adds a critic to estimate values, while GRPO and the REINFORCE-style estimators drop the critic and compute advantages from a group of samples. A reference model exists only when a KL penalty is turned on, to hold the policy near a frozen copy, and KL is off by default. The reward can come from a model or from a plain scoring function. So a run might keep two models alive, or five.

Whatever the count, giving each model its own GPUs would multiply the hardware a run needs. The usual fix is colocation, and it is not unique to verl: put several roles on the same physical GPUs and have them take turns across the phases of the step, so one set of cards serves the whole loop rather than a dedicated set per role.

verl does not place anything on a GPU itself. It describes what it wants and hands the description to Ray's scheduler, so at every layer there are two things to keep straight: the Ray primitive and what it guarantees, and verl's wrapper around it and why the wrapper exists. I'll work one case in detail, eight GPUs on one node, watching GPU 3 where the actor-rollout-ref worker and the critic share space, then widen it to the multi-node, multi-pool layout a real run uses.

### What the layers are

The plan is assembled before any worker exists, in the trainer's setup. For PPO that is [`main_ppo.py`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/verl/trainer/main_ppo.py#L111), where a `TaskRunner` runs as a small Ray driver actor. (`main_ppo.py` is stamped deprecated at this commit, to be replaced by `main_ppo_sync.py` in v0.8.0, but it is still the clearest place to watch the plan get built, and what it builds is what the trainer consumes either way.) You do not hand the `TaskRunner` a GPU layout. You hand it a Hydra config, and it derives two dicts from it.

The first is `mapping`, from each model role to the name of the pool it lives on: `self.mapping[Role.ActorRollout] = "global_pool"`, `self.mapping[Role.Critic] = "global_pool"`, and so on as each role is registered. `"global_pool"` is not special, just the hardcoded name verl gives the main pool. An optional reward model or distillation teacher gets its own (`"reward_pool"`, `"teacher_pool"`). The second is [`resource_pool_spec`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/verl/trainer/main_ppo.py#L162), from pool name to a list of per-node process counts, built from `config.trainer.n_gpus_per_node` and `config.trainer.nnodes`. On one eight-GPU node it is `{"global_pool": [8]}`: one pool, one node, eight GPU processes. Both go to a `ResourcePoolManager`, and together they are the whole plan, which pools exist and how big they are, and which role sits on which. Five layers sit between that plan and a Ray actor pinned to a GPU.

#### ResourcePool: the backend-agnostic description

At the bottom of verl's own abstractions is [`ResourcePool`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/verl/single_controller/base/worker_group.py#L27), which knows nothing about Ray. It holds the per-node process count `_store` (the same `[8]` as the spec), `max_colocate_count` (the co-scheduling budget, how many worker actors Ray will admit onto one GPU), and `n_gpus_per_node`, a hardware constant. `world_size` is a property returning `sum(self._store)`, 8 here. So ResourcePool is a pure description of how many processes spread across how many nodes, in terms any distributed backend could consume. Nothing in it commits a single GPU.

#### RayResourcePool: the Ray reservation

[`RayResourcePool`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/verl/single_controller/ray/base.py#L112) subclasses it and adds the Ray half: `use_gpu`, a `name_prefix` set to the pool name so the reservation shows up identifiably in Ray's logs, and `self.pgs = None`. Those placement groups, Ray's actual reservation, start empty and are created lazily, only when something first asks, which lets the trainer build the whole resource plan in memory before committing any hardware.

A Ray placement group is a reservation Ray locks for you up front. You describe it as a list of bundles. A bundle is a set of resource amounts Ray guarantees can be satisfied together on a single node, something like `{"GPU": 1, "CPU": 3}`, and Ray will not split a bundle across machines. A placement group is those bundles plus a strategy for how they sit relative to each other in space. Once Ray has placed the group, you launch actors into specific bundles and they run on exactly the resources that bundle reserved.

For our `_store = [8]`, [`get_placement_groups`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/verl/single_controller/ray/base.py#L130) builds the bundle from `max_colocate_count`:

```python
bundle = {"CPU": self.max_colocate_count}
if self.use_gpu:
    bundle[device_name] = 1
```

Here `max_colocate_count` is 3, the default this path leaves unchanged, and the CPU count is set to that same number, so the bundle is `{"CPU": 3, "GPU": 1}`: one GPU and three CPU slots. The scheme is one placement group of eight identical bundles, one per process, with strategy `STRICT_PACK`, and a blocking wait on the group's readiness holds until Ray has actually allocated the hardware. When it returns, the GPUs are locked.

Ray's strategies say where bundles may land relative to one another: `PACK` puts them on as few nodes as possible but tolerates spilling, `STRICT_PACK` requires every bundle in the group on one node and fails the reservation if no single node can hold them all, and `SPREAD` and `STRICT_SPREAD` are the mirror image. verl wants `STRICT_PACK` because the eight workers in a pool are a tensor-parallel and data-parallel mesh talking over intra-node NVLink, and splitting them across two machines would route the TP all-reduces over the network and wreck throughput. So RayResourcePool turns "eight processes on one node" into a concrete Ray reservation, deferred until needed, with the intra-node guarantee the mesh depends on baked into the strategy.

#### The fractional GPU

The bundle reserves one whole GPU, and each worker that lands in it asks Ray for only a fraction of one. That fraction is set once, in [`_create_worker`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/verl/single_controller/ray/base.py#L621): `num_gpus = 1 / resource_pool.max_colocate_count`, which is `1/3` here. A fractional `num_gpus` is Ray's own accounting: an actor that asks for a third of a GPU is telling Ray's scheduler that up to three such actors may sit on one card, and the bundle's three CPU slots are the matching half of that gate, since each actor also takes one CPU. Sizing the CPU count and the GPU fraction both at `max_colocate_count` keeps the two budgets in step, so a card has room to admit that many co-scheduled actors.

The part that took me a little bit to understand is that only one actor ever lands in each bundle. verl fuses every role mapped to a pool into a single `WorkerDict` class, then instantiates that class once per GPU in the pool, each instance a Ray actor that holds every role's worker as a plain in-process object. For our one node of eight GPUs that is eight `WorkerDict` actors, one per card. So GPU 3 holds one of them, and it takes a third of the card in Ray's accounting while the other two thirds go unclaimed. That third is a placement count. It does not partition the hardware, so the single actor on GPU 3 can use the whole card's memory and compute.

The `3` is sized for an arrangement verl has grown out of. The comment beside `max_colocate_count` still calls it "the number of WorkerGroups" in a pool and gives the FSDP case as three, "actor_critic_ref, rollout, reward model (optional)" ([`base.py:201-202`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/verl/single_controller/ray/base.py#L201-L202)). Those were once three separate worker groups, three actors sharing a card at a third each. Fusing them into one worker collapsed the three actors to one, and the fraction is what that older arithmetic left behind.

![What occupies one GPU's bundle](/figures/verl-retrospective/colocation-packing.svg)

The fraction was never a memory partition. Ray does not hold an actor to a third of the card's memory, and it does not make the colocated roles take turns. The `1/3` governs placement alone, how many actors Ray will admit. Fitting several models inside one card's memory is handled a layer up, by offloading weights and optimizer state while a role is idle and by the actor and rollout phases not running at the same instant. So the actor-rollout-ref worker and the critic coexist on GPU 3 because they live in one process that reserves a single slice, and verl manages what that process keeps resident.

#### ResourcePoolManager: roles, names, and the indirection

Above the pools sits [`ResourcePoolManager`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/verl/single_controller/ray/base.py#L181), a small dataclass whose `resource_pool_spec` gives each pool name its per-node GPU counts, setting how many pools exist and how big each one is. Its `mapping` records which pool each role is assigned to, and its `resource_pool_dict`, keyed by pool name, holds the actual `RayResourcePool` objects, starting out empty and filling in once the pools are built. It also carries a `max_colocate_count` that defaults to 3, which is where the 3 that propagated down to the fractional GPU comes from.

`create_resource_pool` walks the spec and fills `resource_pool_dict`, one `RayResourcePool` per named pool. Then [`_check_resource_available`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/verl/single_controller/ray/base.py#L223) runs. It sums every GPU Ray reports across the cluster, sums every GPU the spec asks for, and compares the totals, and that is the whole check. It never looks at the per-node distribution, so if you ask for `{"global_pool": [8]}` on a cluster that is really two nodes of four GPUs, the totals match and this passes, and the failure surfaces later, when `STRICT_PACK` cannot find one node with eight free GPUs and the reservation hangs.

[`get_resource_pool`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/verl/single_controller/ray/base.py#L215) resolves a role to its pool object:

```python
def get_resource_pool(self, role) -> RayResourcePool:
    return self.resource_pool_dict[self.mapping[role]]
```

It chains two lookups, role to pool name through `mapping`, then pool name to the pool object through `resource_pool_dict`. Because several roles can map to the same name, the chain can hand back the same pool twice. When the critic and the actor both map to `"global_pool"`, both calls return the same `RayResourcePool` object, the same Python instance, identical by `id()`. That shared identity is the whole mechanism of colocation at this layer. The trainer is about to use these pool objects as dictionary keys, and because the actor's pool and the critic's pool are one object, they collapse to a single key, so everything filed under that pool gets fused onto the same GPUs. Without that shared identity, every role would need its own pool, back to multiplying the hardware a run needs.

The role names come from a `Role` enum, `Role.ActorRollout` to `"actor_rollout"`, `Role.Critic` to `"critic"`, `Role.RefPolicy` to `"ref"`, `Role.RewardModel` to `"rm"`, `Role.ActorRolloutRef` to `"actor_rollout_ref"`, and those strings are the per-role prefixes `spawn` strips off each method name. So the decision of which roles share GPUs lives in one editable `mapping` dict, instead of scattered across the code.

### The build loop

With the pools created, the trainer turns the plan into actors. It builds a routing dict keyed by pool object, one empty inner dict per distinct pool, then for each role the run uses it resolves the pool and files a [`RayClassWithInitArgs`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/verl/single_controller/ray/base.py#L336) under it. `RayClassWithInitArgs` is the deferred-instantiation wrapper: it stores a class plus the constructor arguments to call it with and creates nothing. You describe a worker before creating it because the same description gets instantiated eight times, once per GPU, each with different Ray options, a different bundle index, a different rank, a different actor name, and building the recipe once to replay per GPU beats threading eight near-identical constructor calls through the trainer.

The critic resolves the same pool object, so its description lands in the same inner dict as the actor's under the key `"critic"`, and the routing dict now has one pool key holding two role descriptions. [The loop](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/verl/trainer/ppo/ray_trainer.py#L861) runs once per pool: `create_colocated_worker_cls` fuses every role description under that pool into a single `WorkerDict` class, `RayWorkerGroup` instantiates that class once per GPU in the pool, and `spawn` hands back one worker-group view per role with the prefix stripped, so `all_wg["critic"]` and `all_wg["actor_rollout_ref"]` are separate handles onto the same actors.

We have already traced what [`create_colocated_worker_cls`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/verl/single_controller/ray/base.py#L986), the `WorkerDict` wrap, the `{role}_` prefix, and `spawn` do to a method call, so I will not redo it. One status note before moving on: the colocation builder carries a `# deprecated, switching to FusedWorker` stamp, its replacement written and tested but not yet wired into any trainer, the same half-finished shape as the DataProto-to-TensorDict migration. On the placement side it comes to this: the fusion is why two roles live in one Ray actor on one GPU, the prefix is how their colliding method names (both the actor and the critic define `save_checkpoint`) stay distinct inside that actor, and `spawn` is what gives the trainer back the per-role handles it dispatches through.

`RayWorkerGroup` is where the placement groups finally get created. It asks its `RayResourcePool` for them, the pool builds them on that first call and caches them in `self.pgs`, and the worker group schedules into them, with `STRICT_PACK` because `bin_pack` defaults to `True`. It loops the bundles calling `_create_worker` per GPU, and each call injects the rank and world-size environment variables, then calls the `RayClassWithInitArgs` recipe with a `PlacementGroupSchedulingStrategy` pinned to that GPU's bundle index. That strategy is the Ray primitive that pins an actor to exactly this bundle of this placement group. Bundle index 3 is GPU 3, so the worker that becomes rank 3 lands on GPU 3, and that determinism is what lets the mesh treat rank 3 as a known NVLink neighbor of ranks 2 and 4. The `1/3` GPU fraction rides along in the same options, so the rank-3 actor reserves a third of GPU 3.

That closes the loop back to the dispatch machinery. The eight actors the dispatch machinery fans out to are eight `WorkerDict` instances, one per GPU in `global_pool`, each holding the fused actor-rollout-ref worker and the critic as two in-process objects, each instance reserving a third of its card. When `actor_wg.compute_log_prob(data)` chunks the batch into two DP shards, the workers those shards reach are these eight actors, placed by this loop.

### Scaling up

That covered one pool on one node. The full layout in `main_ppo.py` assigns the roles to pools. The actor side is fused: unless LoRA forces a [dedicated reference model](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/verl/trainer/main_ppo.py#L140), the reference policy lives in the same worker class as the actor and rollout, registered under `Role.ActorRolloutRef` and mapped to `"global_pool"`. The critic maps to `"global_pool"` too. The reward model is the role with a choice, mapping to a separate `"reward_pool"` when its resource pool is enabled and to `"global_pool"` otherwise, and the spec grows a second pool only when that flag is set.

![How the resource-pool machinery places workers, shown across two nodes](/figures/verl-retrospective/machinery-map.svg)

So two nodes of eight GPUs become `global_pool: [8, 8]`, which becomes two placement groups, one `STRICT_PACK`ed onto each node. On every one of those sixteen cards one Ray actor holds the same fused stack, the actor-rollout-ref worker and the critic together, reserving a third of the card. If the reward model gets its own pool, a separate `reward_pool` runs it on its own GPUs and the actor side keeps `global_pool` to itself. The role-to-name-to-object indirection is what routes this: every role naming `"global_pool"` resolves to one object and collapses to one key, so each card gets one fused actor, and the reward model naming `"reward_pool"` resolves to a different object and its own GPUs.

One small note about this path: the [reward model](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/verl/trainer/main_ppo.py#L197) is registered only in the pool mapping, with no worker, its scoring running through a separate reward-loop manager, so no reward worker is colocated on the cards.

## The tooling the fork required

None of this was what I came to verl to do. Before I could touch the parts I cared about, I had to make the fork livable, and two pieces of that turned into real work: getting the install to behave, and building tests I could trust.

### Packaging

What I wanted out of packaging was an install that just works on a fresh machine, and a clean line between the core every run needs and the optional pieces a particular run pulls in. verl reaches for exactly that, sorting its dependencies into ten extras groups in `setup.py` ([`setup.py:62-73`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/setup.py#L62-L73)). The grouping is the right idea, but in practice it leaks at every join I leaned on.

The first thing that surprised me is that `torch` is not in `install_requires` at all. The core dependency list runs from `accelerate` down through `tensordict` and `transformers` ([`setup.py:26-45`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/setup.py#L26-L45)) with no PyTorch in it. The only pinned `torch` in the whole file lives inside the `sglang` extra, as `torch==2.9.1` ([`setup.py:57`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/setup.py#L57)). So the framework whose entire job is distributed training on GPUs leaves its tensor library to be satisfied by something else in your environment, which is a defensible choice for a project that has to coexist with whatever vLLM or a base image already pinned, and also a thing you have to know before a clean install does what you expect.

Then the same dependency line shows up three times. `tensordict>=0.8.0,<=0.10.0,!=0.9.0`, character for character, in the core list ([`setup.py:40`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/setup.py#L40)), in the `vllm` extra ([`setup.py:52`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/setup.py#L52)), and in the `sglang` extra ([`setup.py:55`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/setup.py#L55)). Three copies of one constraint means three places to edit when the upper bound moves, and three chances for them to drift apart.

The split I actually wanted to rely on is contradicted by a second file. `requirements.txt` and `setup.py` disagree about what counts as required. `uvicorn`, `fastapi`, `latex2sympy2_extended`, `math_verify`, `liger-kernel`, and `TransferQueue==0.1.7` all sit flat in `requirements.txt` as things every dev install pulls. In `setup.py` the story is different: `liger-kernel` is gated behind the `gpu` extra, `math-verify` behind `math`, and the rest are not in the core list at all. So the answer to "what does verl need to run" depends on which file you read, and the two were maintained by different hands at different times.

The example I kept coming back to is `pyext`. It is declared in the `prime` extra ([`setup.py:48`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/setup.py#L48)), and it is still imported at the top of the PRIME code scorer, `from pyext import RuntimeModule` ([`testing_util.py:36`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/verl/utils/reward_score/prime_code/testing_util.py#L36)). The tests that exercise that path are skipped with the reason `"pyext not compatible with python 3.12"` ([`test_sandbox_on_cpu.py:117,175`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/tests/utils/reward_score/test_sandbox_on_cpu.py#L117)), and the repo's own install script says the same thing out loud, `echo "pyext is lack of maintainace and cannot work with python 3.12."`. So here is a dependency that the project knows is unmaintained, knows breaks on the Python version everyone is now using, has disabled its tests for, and still imports at module load on a code path that is wired into reward scoring. Nobody has removed the import, and the code path that needs it cannot run on current Python, so it sits in an in-between state that a clean core-versus-optional boundary is supposed to make impossible.

There is also genuine age in the core list. `pylatexenc` ([`setup.py:37`](https://github.com/verl-project/verl/blob/9f73954a87e247de4c31bc2f5222969e395b2904/setup.py#L37)) is a hard core dependency whose last stable release predates most of what verl is built on. I am not going to do a package-by-package audit here, partly because the repo has moved under me since I forked and several of these specifics have already shifted, and a then-versus-now table would be stale before it was published. The structural shape is what stays true: the core-versus-optional split exists on paper, and in practice the boundary leaks in every direction. Cleaning that up was the first tax the fork charged me, before I had changed a single line of the parts I came to change.

### Tests

With the install behaving, the thing standing between me and the refactors was a test suite I could rely on.

You cannot refactor a system this size by reading carefully and hoping. I needed to know, after every change, whether the rest of the codebase still did what it did before, which means tests of a specific kind: automatic, launchable on a node or two without me babysitting them, broad enough to cover the surface I was about to disturb, and honest about what broke when something broke. verl has a large test suite, so the raw material was there, just not in a state I could lean on.

A lot of the work was unglamorous repair. Stale `skipif` guards pinned to `tensordict` versions years out of date, so whole files quietly did nothing. Tests that wrote scratch files into the working directory and deleted them by hand, which leaves litter behind on the first failure and races against itself when you run things in parallel, moved over to pytest's `tmp_path`. Duplicated helpers like `union_tensor_dict`, which existed in two files with two behaviors, collapsed to one. A good chunk of the end-to-end coverage was not Python at all: seventeen-odd shell scripts under `tests/special_e2e/` that drove a training run through environment variables and `bash`, which a CI system cannot schedule, cannot shard, and cannot report on at the granularity of a single failing assertion. Converting those into real pytest tests meant rebuilding their Ray setup as fixtures, replicating the production runtime environment so a test would not silently run with the wrong NCCL settings, and replacing `torchrun` entry points with `mp.spawn` workers that bring up and tear down their own distributed process group. The recurring lesson there was that the shared logic wanted to live in one `conftest.py` and be reused, not be copy-pasted into every file.

The piece I am actually glad I built is the GPU-aware test scheduler, and the problem it solves is specific to this kind of codebase: the test files do not all want the same amount of hardware. Some need one GPU, some need two, the end-to-end ones want eight or a whole node. Run them one after another and a single-GPU test sits on one card while seven idle which is an expensive way to wait. What I wanted was to mark each test file with the number of GPUs it needs and then pack the files onto whatever cards were free, so that when a two-GPU test finished, the gap it left got filled immediately by the largest pending test that fit.

I looked for something off the shelf first. `pytest-xdist` is built around a fixed pool of worker processes, which is the wrong model when the unit you are scheduling is GPUs and different jobs claim different numbers of them. `pytest-isolate` was closer but young, and its interaction with Ray was untested, which matters a lot here because the tests themselves call `ray.init()`. Ray could in principle orchestrate the whole thing, except Ray-inside-Ray, a scheduler built on Ray launching tests that each start their own Ray, is its own source of trouble. So I wrote a standalone scheduler using asyncio to manage test subprocesses: collect the test files and their GPU requirements through an embedded pytest plugin, then greedily assign the largest job that fits onto the free cards, each test file running as its own subprocess with `CUDA_VISIBLE_DEVICES` set to the cards it was given.

The first version had the architecture right and the details wrong, and the way the details were wrong taught me more than the architecture did. The clearest one: when a test timed out, the scheduler killed the pytest process and considered the card freed. But these tests spawn children, Ray workers and torch distributed processes, and on Linux killing the parent orphans the children, which keep their CUDA contexts alive and the GPU memory allocated. The next test scheduled onto that "free" card would hit out-of-memory or see the wrong device count. The fix was to launch each subprocess as its own process group, with `start_new_session=True`, and kill the whole group on timeout, so the children go down with the parent.

That shape, a fix that was correct in isolation but wrong against code written under the old assumption, kept recurring. Adding a collection timeout introduced a temp file that leaked, because the `sys.exit` in the new timeout path jumped over the `finally` that did the cleanup. Adding a periodic heartbeat for long-running tests printed the status twice, because the heartbeat looped back to the top of the scheduler where a status print already lived. Each one was a small bug, and each one came from the same source: a new code path firing a side effect that was correct for every path that existed before it. The interrupt handling had a subtler version of the same problem. The first cut caught Ctrl+C, killed the processes, and waited for the tasks to finish, but never collected their results, so interrupted tests just vanished from the final summary instead of being reported. Later I learned that under recent Python, `asyncio.run` installs its own signal handler and the exception that actually reaches the coroutine is `CancelledError`, not `KeyboardInterrupt`, so the cleanup I had written was not guaranteed to run at all. Both got fixed, the second by catching both exceptions.

The bugs that survived every round of reading were environmental, and those are the ones I think about. The scheduler set `RAY_ADDRESS=""` in the subprocess environment, meaning to say "do not connect to any existing cluster." That reads as reasonable, and it passed several rounds of me staring at it. On a real machine it hung indefinitely, because Ray does not treat an empty string as "no cluster", it treats it as an address and tries to connect to it. The fix was to remove the variable from the environment entirely rather than set it empty, so Ray falls back to starting its own local cluster. The other one of this kind cost a day: on a host where `TMPDIR` pointed at an NFS mount, Ray's worker startup hung, because Ray puts its session files under `TMPDIR`, its workers grab file locks during startup, and that NFS mount delegated locking to a network lock manager that fell over under a hundred-plus workers all locking at once. None of that is visible in the source. You can only find it by running the thing on the machine it will actually run on, which is the whole reason I needed the scheduler to begin with, and it is why a test harness you can launch on a real node and trust the output of was worth building before I touched the code I came to verl to change.

## Why I stopped

The refactors I cared about were the orchestration layer and the data layer. Both turned out to be harder than expected.

The dispatch cleanup is the smaller of the two. Moving the config off the magic function attribute and onto the class as ordinary, validated data does not change how dispatch runs. The metadata is carried through `@register` on every worker method, read back during `WorkerGroup` construction, and assumed in that shape by the colocation builders, the `spawn` prefixing, and the fused-worker path. Changing how it is carried meant editing the decorator, the construction scan, and every worker that depended on the old contract together. The data layer was the same shape at a larger scale. DataProto is the driver-side type and TensorDict is the worker-side type, and the conversion sits at every boundary, so pulling DataProto out reaches the trainer hierarchy, the worker mixins, the data loader, the protocol tests, and the serialization paths. verl is already doing that migration in pieces for exactly this reason. I had told myself I could land at least one of these chunks on my own. But, I could not, the surface each one touched was most of the framework.

That much is workable if the ground holds still. But, verl is in active development, with new releases and fixes landing almost daily. So a refactor in `reverl` was never finished against a fixed target. Whatever I changed in the orchestration or the data layer had to keep working the way upstream now did it, or get redone once a release moved the same files. The DataProto migration I was reading my way into kept advancing upstream while I worked, so my branch both went stale faster than I could change and the cost of staying in sync was larger than the refactor, which I wanted to work on.

Added up, maintaining a divergent fork is not really worth it besides for my satisfaction, but even that was not given since I had to keep in sync all the time.

## Where this goes

Reading verl this closely left me with a working picture of how an RL post-training system fits together, and a set of opinions about the parts I would build differently. Some of those I sketched out here, like moving the dispatch metadata off that hidden function attribute and onto the class. The rest are going into an orchestration layer of my own, which I will write more about in an upcoming post.

## Bonus: the NCCL bootstrap hang

The opening listed chasing NCCL bootstrap hangs among the costs of the fork. Here is the one that taught me the most, kept as a bonus because its shape generalizes past verl to any multi-GPU stack on data-center networking.

The symptom was a hang with nothing to grab onto: no error, no timeout, no crash. A two-GPU correctness test brought up two Ray worker actors, each with one GPU. Both loaded the model, both passed `torch.distributed.barrier()`, and both then stopped inside the `FSDP()` constructor called with `sync_module_states=True`.

What made it tractable was learning that PyTorch's distributed stack is three separate communication systems, each with its own network setup:

- The store. `init_process_group` with the default `env://` method makes rank 0 start a TCPStore key-value server at `MASTER_ADDR:MASTER_PORT`, and every other rank connects to it to exchange metadata and the NCCL unique id. verl derives `MASTER_ADDR` from Ray's node IP, which on this host had already been pinned to `127.0.0.1`, so the store was reachable.
- Gloo, the CPU half of verl's composite `cpu:gloo,cuda:nccl` backend, which carries CPU-tensor collectives. It rides the store at `MASTER_ADDR`, so the CPU side had a clear path over `127.0.0.1`.
- NCCL, the CUDA half, which is a separate library with its own bootstrap. Rank 0 generates a unique communicator id that carries a bootstrap IP and port, broadcasts it through the store, and every rank connects to that bootstrap address to trade transport-level details. The detail that matters: NCCL chooses that bootstrap IP itself, by enumerating the host's network interfaces, and never consults `MASTER_ADDR`.

So the barrier completed and the first NCCL collective hung, and that gap is the whole story. `FSDP(..., sync_module_states=True)` broadcasts rank 0's parameters to the other ranks, which is the first NCCL operation in the run and the one that lazily initializes the communicator and triggers the bootstrap. On this host, NCCL's interface enumeration picked a bonded VLAN interface whose IP was not self-routable: the machine could not open a TCP connection to its own bonded address. Rank 0 listened for the bootstrap on that address, the other rank's `connect()` blocked until the multi-minute TCP timeout, and the run simply sat there. Gloo was fine the whole time because it spoke over `127.0.0.1`. NCCL hung because it had picked its own interface and never looked at `MASTER_ADDR`.

The method that got me there is the part worth keeping:

- Find the layer. A distributed hang is stuck in exactly one place: Ray's own startup, the store and Gloo, NCCL's bootstrap, or NCCL's data transport. Each detects its network independently, so a fix at one layer does not carry to the next. Pinning Ray's node IP earlier in the same saga had gotten `ray.init` healthy and left NCCL hanging, for exactly this reason.
- Shrink the reproduction. The real test took minutes to reach the hang. Two Ray workers reproduced it in seconds: a bare `torch.distributed.barrier()` passed, while `torch.distributed.all_reduce(torch.ones(4, device="cuda"))`, a collective on a CUDA tensor that routes to NCCL, hung. That puts the fault on NCCL, with the model, FSDP, and the test code all out of the picture.
- Turn on NCCL's own logs. `NCCL_DEBUG=INFO` with `NCCL_DEBUG_SUBSYS=INIT,NET,BOOTSTRAP`, set inside the worker process. NCCL reads them when it initializes the communicator, which the first collective triggers, not at import, so they have to be set in the process that runs the op. Ray captures the worker's output under `/tmp/ray/session_*/logs/`, so look there, not in the pytest output. A healthy run prints a bootstrap line like `Bootstrap: Using lo:127.0.0.1<0>` and, later, an `Init COMPLETE` line. The hung run reached neither.

The fix is one environment variable, `NCCL_SOCKET_IFNAME=lo`, which forces NCCL to bootstrap over loopback. That is safe on a single node because every rank is on the same machine, so `127.0.0.1` reaches all of them, and the bootstrap socket only carries the small init handshake. NCCL's real data path, NVLink between the GPUs on the box and InfiniBand RDMA between boxes, is selected separately and stays untouched, so loopback bootstrap costs nothing at transfer time. The harness pins `GLOO_SOCKET_IFNAME=lo` next to it for consistency.

Two things keep that from becoming a footgun. The harness sets the variables only when it detects that the host's outbound IP is not self-routable, and only when they are not already present, so a genuine multi-node run can point `NCCL_SOCKET_IFNAME` at an interface that routes between nodes and override the default. And the detection has a cost worth respecting: the probe opens a TCP socket with a one-second timeout against the host's own IP and waits for it to fail, so the test for "are these already set" has to run before the probe, not inside it. The shipped guard is:

```python
_ROUTING_VARS = ("RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME")
if not all(v in os.environ for v in _ROUTING_VARS):
    if not _node_ip_is_self_routable():
        for _var, _val in zip(_ROUTING_VARS, ("0", "lo", "lo")):
            if _var not in os.environ:
                os.environ[_var] = _val
```

Set before `import ray` and inherited by every Ray worker and pytest subprocess, the `all(...)` dict lookup means a child that already carries the variables skips the one-second probe entirely.

The lesson I take out of it: a distributed hang that only appears on the cluster is almost always the network, and specifically an IP that does not route back to itself. A modern training stack has several components that each decide independently which interface is the host's, and on a bonded or VLAN-isolated data-center NIC they can each pick wrong, one at a time, which is why the same root condition gets fixed in more than one place. The durable defense is to detect the condition at runtime and pin the interface to loopback when the host cannot route to itself, so a new machine with different networking does not bring the hang back.

For the record, the box was a single node of eight H100s on a bonded VLAN interface whose IP did not route back to itself, with NCCL `2.27.5+cuda12.9`, PyTorch `2.7`, and Ray `2.54.0`. Loopback bootstrap is a single-node measure. A real multi-node job needs `NCCL_SOCKET_IFNAME` pointed at an interface that routes between the nodes.
