+++
title    = "A Comprehensive Guide to Python Project Management and Packaging: Concepts Illustrated with uv - Part II"
date     = "2024-11-20T19:46:43+00:00"
draft    = false
categories = ["Python"]
tags       = ["popular", "Packaging", "uv"]
+++


In the [first part](https://reinforcedknowledge.com/a-comprehensive-guide-to-python-project-management-and-packaging-concepts-illustrated-with-uv-part-i/) we delved into the nitty gritty details of initializing and managing the dependencies of a Python package or project. This covered what is a Python package as opposed to a plain project (and other nomenclature), what is defined in the standard (e.g., `project`, `build-system`, `tools` and `dependency-groups` tables in `pyproject.toml` through the PEPs [518](https://peps.python.org/pep-0518/), [621](https://peps.python.org/pep-0621/) and [735](https://peps.python.org/pep-0735/)) and what are some common features that tools provide on top of the standard (e.g., lock files, editable dependencies).



In this second part, we'll delve into the building and publishing part of the workflow. We're going to follow a similar structure as in the first article where we first present the high-level overview of the concepts and we illustrate them using `uv` then we cover the in-depth details by studying the related PEPs and specifications.  
  
Since we're covering the build part in this article, we'll also take peeks and explain parts of the code of the build backend that we're going to use. It's not necessary to follow along, I just like to unravel the magic behind these seemingly complex tools.  
  
As mentioned in the first part, the terms used in this domain come with some ambiguity. What some would call building a Python package, others would simply call packaging a project. Though I prefer the first way of saying, both ways refer to using (or not) some metadata to transform your code into something that's portable, that can be published, and that others can get and use easily.  
  
We'll only focus on pure Python packages in this article and we won't delve into building projects with C extensions or other compiled languages. This choice respects the structure of the first part and aligns with my philosophy as well, which is to deep dive into every core concept that we encounter along our way of understanding what we want to understand. And including complex projects that contain C code would require me to explain in depth C code compilation, what's a shared object etc. That'll make the article unnecessarily complex when it comes to understanding Python packaging. Once those fundamentals are clear, it becomes easier to build upon them and extend to projects requiring extensions in other languages.  
  
We'll also only focus on packaging pure Python libraries and applications intended for distribution via standard Python packaging tools like `pip`. This means we're assuming that the end users are developers or users who already have a Python environment set up and are comfortable installing packages using tools like `pip`. We won't delve into methods for packaging Python applications into standalone executables or containers that bundle the Python interpreter and all dependencies. Tools such as `PyInstaller` and `Docker` containers among others allow developers to distribute applications to environments where Python may not be installed. But this is beyond the scope of this article. Our goal is to understand the standard and core concepts behind the standard formats for distributing Python packages and how the standard came to be, which means we'll focus on source distributions and wheels and how to build and publish them effectively. Which means we'll also not cover the deprecated and old **egg** format which was introduced by `setuptools` in 2004 and which is replaced by the wheel format as per [PEP 427](https://peps.python.org/pep-0427/). Unlike wheels, eggs did not have an official standard specification and served both as a distribution and a runtime installation format. As of August 2023, PyPI rejects egg uploads, and the Python community has largely moved towards using wheels for package distribution.  
  
For the rest, I refer you to <https://packaging.python.org/en/latest/overview/#packaging-python-applications>.  
  
Let's get started!  
  
**PS:**  
- For an introduction to this topic in Turkish, you can read the following post: <https://www.sglbl.com/2025/01/uv-ve-pyprojecttoml-ile-python-projesi.html>



# uv build



Let's initialize a folder as a package using `uv`, `uv init --package my_package`. If you don't know what the `--package` does, refer to the first part, or even better read the manual for it 😁.  
  
If you open the `pyproject.toml` you'll notice a `build-system` table which contains two keys, `requires` and `build-backend`. `uv` [doesn't have its own build backend yet](https://github.com/astral-sh/uv/issues/3957) (`uv` version 0.5.1 (f399a5271 2024-11-08)) so it defaults to [hatchling](https://pypi.org/project/hatchling/). If you want to configure some of `hatchling`'s options to tailor it to your project, you'll have to install [hatch](https://github.com/pypa/hatch) and use the `hatch build` command's options. `hatchling` is the build backend of `hatch`. There are other [build backends](https://packaging.python.org/en/latest/guides/tool-recommendations/#build-backends) that you can use.



## Build systems






### High-level overview



**Related PEPs**: [517](https://peps.python.org/pep-0517/), [660](https://peps.python.org/pep-0660/), [610](https://peps.python.org/pep-0610/)



The **build system** is the combination of a build backend and a build frontend and whatever configuration or workflow management that's required to make the frontend and backend communicate.  
  
The **build backend** is the library or tool that will transform your code from a source tree, which is your source code + the `pyproject.toml` with some metadata that specify the build backend (more about that in the section on package formats), into a package, either a source distribution (sdist) or a built distribution (wheel). The backend does the actual heavy lifting. In our case it's `hatchling`.  
  
The **build frontend** is the user-facing part of the build system, it takes a source tree and delegates its building to its build backend. Since we're using `uv build`, our build frontend is `uv`.  
  
It's standard terminology across software engineering. An analogous situation would be the Docker CLI that acts as a frontend while `dockerd`, the Docker daemon, is the backend that does the heavy lifting, and the whole system would also contain Docker Hub and Dockerfile etc. You can also think of the Git porcelain commands as the frontend while the plumber commands are the backend and the whole system would contain the `.gitignore`, `.gitconfig` etc.  
  
**Notes**:   
- Though the build backend is the heavy lifting component and the build frontend is the user-facing component, this latter one can make some decisions beyond the user as part of its default behavior on what to ask from the build backend, so **it's not just parsing arguments and calling the backend**.  
- Build frontends are also responsible for setting up an isolated (not mandatory but highly recommended by the PEP) Python environment in which the build backend will run and in which the build requirements will be installed.



### Details



Before we delve deeper into what standard compliant build systems look like, let's first check what issues the Python community suffered from before the current standard.  
  
We'll talk a lot about [PEP 517](https://peps.python.org/pep-0517/), and a little bit about [PEP 660](https://peps.python.org/pep-0660/) for editable installs, because that's where the interface for build backends and recommendations for build frontends are established. But we have to understand that, though [PEP 517](https://peps.python.org/pep-0517/) brought the standardization together, it'd have been hard to achieve (in my opinion) without the wheel format from [PEP 427](https://peps.python.org/pep-0427/). Let's see why.  
  
In the following we'll mention a lot the words source distribution (or sdist for short) and wheel, we'll dive deeper in them in the upcoming sections, but you can consider them as output formats when you build your package. So a build frontend will take the source tree and ask the build backend to build it as a source distribution for example.



#### The need for standardization



We talked about that in detail in the [first](https://reinforcedknowledge.com/a-comprehensive-guide-to-python-project-management-and-packaging-concepts-illustrated-with-uv-part-i/#What-did-we-have-before-pyproject.toml) part, about how the existing `distutils` was insufficient and how `setuptools` caused a lot of headache with circular dependency problems and other issues.  
  
I'll focus here on the three main issues mentioned in the [PEP 517](https://peps.python.org/pep-0517/).



- The first issue was missing important features. 
  - **Lack of usable build-time dependency declaration**: Both `distutils` and `setuptools` didn't offer satisfying ways to declare build-time dependencies, which are all the dependencies that you need to build your project but don't need in your project. We talked about that in-depth in the first part of this article so I won't get into it here, but `setuptools` offered a way to do that with `setup_requires` which caused a lot of issues since different build tools couldn't necessarily process that, and to process the `setup.py` itself you needed the `setuptools` dependency which you can't know before processing the `setup.py`. This led to manual management of build dependencies and inconsistent builds.
  - **Lack of autoconfiguration**: support for automatic discovery of environment configurations and platform-specific features and needs was very minimal which led developers to write platform-specific code in the `setup.py` which is not optimal since it burdens the developers, creates fragile and hard to maintain build scripts.
  - **Lack of niceties like DRY-compliant version number management**: ([DRY](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself): Don't Repeat Yourself) There was no built-in mechanism for version management, like synchronizing it across the `setup.py` and `__init__.py` for example or automatically discovering it etc. Which is a hassle since you have to remember that by yourself and make the changes manually every time.
- **Difficulty in extending** `distutils` and `setuptools` and **difficulty in using anything else**: I also talked about that in the first part, but the idea is that these tools (especially `setuptools`) were not much other-tool-friendly tools. It was hard to write a build tool that you can extend with `setuptools` as a plugin for example since it did things so uniquely. There was a time where `setuptools` was a requirement for `pip` since it was so hard to install or build packages without it, mainly due to the `setup_requires`. But `setuptools` came with a wide array of problems and on top of that, requiring `setuptools` halted the progress of other build tools. But obviously with the standardization and evolution of other build tools this was abandoned.



#### How the issues were solved



In the first part we said that the main contribution (at least in my opinion) of [PEP 518](https://peps.python.org/pep-0518/) was to move from having to execute code through the `setup.py` to having to declare data in the `pyproject.toml`. Which solved all the issues mentioned and introduced standardization.  
  
This is still the case here, but the main contribution (again, in my opinion) of [PEP 517](https://peps.python.org/pep-0517/) is **decoupling the build system from the installation** and **standardizing the interface for build backends**.  
  
And one of the reasons why we can do that are **wheels**. Thanks to them, build systems' primary role became producing wheels and source distributions that comply with the standards, while the complexity of the different installation environments was handled by installation tools like `pip`, which will select the appropriate wheel or fallback to build the package from source if none was found.  
  
Decoupling ensures that we can use other tools than `distutils` and `setuptools` (which is the issue that the PEP aims to solve).  
  
But, just refocusing the responsibilities of build systems and separating them from installation tools doesn't automatically ensure decoupling. The interface of build backends must be standardized so there is consistency in using them and not only that, there must be a standard for the wheels and the sdists so that no matter the build system, the output will be "universal" and directly usable by any installation tool.  
  
So [PEP 517](https://peps.python.org/pep-0517/) **standardizes the source tree** by centering it around the `pyproject.toml` and **the source distribution format**, **defines a build backend interface** with mandatory `build_wheel` and `build_sdist` hooks and optional `get_requires_for_build_wheel`, `prepare_metadata_for_build_wheel` and `get_requires_for_build_sdist` hooks and **clarifies the responsibilities of build frontends**.  
  
This allows for an efficient decoupling of build systems and installation tools which solves the aforementioned issues.  
  
We'll now explore the standard for the build backend, the build frontend and the source tree because that's what the build system interacts with.



#### Source trees



##### Standard



A **source tree** is the directory structure containing all the source files of a Python package, typically obtained from a version control system (VCS) checkout. It's everything. It's important to remember that because source distributions, built from source trees, will contain what the source tree contains, but wheels generally don't contain test folders and documentation etc. The source distribution won't contain what is ignored by the VCS and that's why it's important to mention the VCS here as well.  
  
[PEP 517](https://peps.python.org/pep-0517/) standardizes source trees by including a `pyproject.toml` at the root of the source tree. It also extends the `build-system` table defined in [518](https://peps.python.org/pep-0518/) by the `build-backend` key. It **specifies the Python object that will perform the build**. It follows the `module:object` syntax, similar to entry points. You can also just use `module`.  
  
The specific rules are: `entry_point = module_path (':' object_path)?` (here `entry_point` is the `build-backend` string). Which means that the `build-backend` string is a `module_path` optionally followed by a colon and an `object_path`. Where `module_path = identifier ('.' identifier)*`, which means that a `module_path` is one or more identifiers separated by dots, and same for `object_path`, `object_path = identifier ('.' identifier)*`. And an identifier is: `identifier = (letter | '_') (letter | '_' | digit)*`, so an identifier starts with a letter or underscore, followed by letters, digits, or underscores.  
The PEP correctly presents the [BNF notation](https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_form), I changed the order here for your understanding.  
  
In our case, `build-backend = "hatchling.build"`, we only have a `module_path`.  
  
It's important to understand that `build-backend` is similar to an entry point but it's not the same. The `build-backend = module_path` means that **the build frontend will perform this**:  
`import module_path  
backend = module_path`  
  
In our case, the build frontend imports `hatchling.build` and assigns it to `backend`. It's different from the `hatcling` CLI entry point: `hatchling = "hatchling.cli:hatchling"`.  
  
Then there is the `requires` key which is the list of build time dependencies.  
  
In our case, `requires = ["hatchling"]`. So what the build frontend does is, it installs `hatchling` and then imports `hatchling.build` and assigns it to `backend`.  
  
The PEP also extends the table with an optional key `backend-path` which is a list of directories, relative to the project root, to add to `sys.path` when loading the backend used for in-tree build backends. As we'll see in the next section, this is a prohibited behavior for normal build backends. We'll talk a bit about in-tree build backends later on.  
  
**Note**:  
The concept of source tree is standardized along the `pyproject.toml` file but you can build projects without having it or without having a `build-backend` key. The behavior in these cases is legacy so I won't get into that, but feel free to refer to the documentation mentioned at the beginning of the section.



##### Standard for tools wrt source trees



- **Suggestion**: Import the module path without the source directory
  - The [PEP 517](https://peps.python.org/pep-0517/) says something important "When importing the module path, we do *not* look in the directory containing the source tree, unless that would be on `sys.path` anyway".
- **Responsibilities**:
  - Build frontends *must* import the build backend declared in `build-backend` without including the source directory in `sys.pat`h, unless the source directory is already in `sys.path`.
  - Build backends *must* not rely on being imported from the source directory, unless they're explicitly configured as such with the `backend-path`. This configuration is for *in-tree backends* (you probably have guessed it with the name).
- **Gains**:
  - This **decouples the build backend from the source tree** (again, unless we explicitly configured it to do so, and you'll see in the section for in-tree backends that it's a very special case), which totally makes sense. We don't want unintended imports or behavior due to including the source directory in `sys.path`, and so by eliminating this external factor and by importing and running the build backend in an isolated environment we ensure **consistency**.
  - It also **avoids** the risk of importing and running modules from the project's source code while importing the backend module and unexpectedly running **malicious code** if present in the project for example.


- **Suggestion**: Prohibition of cycles in build requirements
  - "Project build requirements will define a directed graph of requirements"
- **Responsibilities**:
  - Build backends *must* ensure that the dependencies in`requires`do not create a circular dependency issue.
  - Build frontends *should* resolve the build requirements and check for cycles.
  - Build frontends *should* refuse or terminate the build in the presence of a circular dependency in the build requirements.
- **Gains**:
  - The gains here are clear, the idea is to avoid impossible builds, prevent errors and save time.
- **Notes**:
  - The responsibility of the build backend for ensuring there is no circular dependency in the declared build requirements is an interpretation of mine because the PEP says "This graph MUST NOT contain cycles" and "front ends MAY refuse to build the project". So I guess the first recommendation is directed towards build backends while the second is less restrictive and is for build frontends which will just run quick checks in order to prevent errors and save time.


- **Suggestion**: Use of wheels to avoid deeply nested builds
- **Responsibilities**:
  - Build frontends *should* use wheels of build requirements when they're **available** **and** when it's **practical**.
  - Build frontends *may* provide modes where they ignore wheels and build from source instead.
- **Gains**:
  - I guess it's a question of both **efficiency** since by using wheels, which are pre-built binary distributions, we avoid the need to build dependencies from source and **simplicity** because we don't have to take into account all of the dependency trees for each build requirement and having to match them etc.
  - And build frontends might sometimes not use wheels for different reasons like wheels not being available for that platform or because the wheels are not distributed through a trusted channel for the organization etc.
- **Note**:
  - The PEP also imposes a responsibility on projects. Since it's not about build backends or frontends directly I chose to omit it, but mainly what it says is that projects *must not* assume that publishing wheels will resolve cycles in the dependency graph. This is very specific (in my opinion) to use cases where either you were able to build a wheel even when having a requirement cycle in your build dependencies, maybe through manual intervention or something else, or your project relies on itself to build itself. And since build frontends might circumvent the wheel, then you must not assume that just by providing a wheel your users (if they use your package as a build requirement) won't encounter a problem. The PEP offers a solution for the self-hosting backends by declaring themselves as *in-tree backends*.
  - Obviously wheels avoid deep nested builds because now you just have the wheels as the requirements instead of having to reason about their dependency graphs.



**Note**s:  
1 - When it comes to the constraints on build requirements, I think the PEP encompasses both the dependencies specified in the `requires` key of the `build-system` table but also the dependencies brought on by the optional hooks `get_requires_for_build_wheel` and `get_requires_for_build_sdist`. It's not explicitly mentioned so take this with a grain of salt, but it seems reasonable. If you were to build a wheel directly without having to build an sdist, then the `get_requires_for_build_sdist` isn't called so I guess the build backend only has to verify that there is no requirement cycle in the extended build dependencies list from `requires` and `get_requires_for_build_wheel`.  
2 - When it comes to circular build requirements, since build frontends might build the requirements from source, it also means that there should be no circular dependency issue in or between the dependencies of the build requirements themselves.



#### Build system interface



In order to keep things generic for the purpose of generality, let's say the `build-system` table in your `pyproject.toml` looks like:  
`[build-system]  
requires = ["some-build-backend"]  
build-backend = "some_build_backend.build"`  
  
This tells any build frontend that the project requires `"some-build-backend"` to build.  
  
There's no need to do something like:  
`[build-system]  
requires = ["some-build-backend", "some-other-dependency"]  
build-backend = "some_build_backend.build"`  
  
Because we have talked about that in the source trees. So without loss of generality, we can keep it like the former. Except if for needs of clarity I change that in a particular sub-section but then I'll mention it.  
  
At the most basic level, we want build backends to do two things, build a source distribution and build a wheel. So we'll start with that.



##### Building a source distribution



Build backends ***must*** implement a `build_sdist` hook, `build_sdist(sdist_directory, config_settings=None)`, that ***should*** create a `.tar.gz` source distribution and ***must*** be placed in the specified `sdist_directory`.  
The PEP focuses on the gzipped tarball format because it's the most used one for sdists.  
  
Build backends ***must*** also output the basename of the created sdist file as a Unicode string.  
  
The generated tarball ***should*** follow the POSIX.1-2001 pax tar format to ensure compatibility and proper handling of UTF-8 filenames.  
  
If the backend cannot build an sdist due to missing dependencies or other issues, it ***should*** raise an `UnsupportedOperation` exception. This signals to the frontend that it may need to adjust its process, possibly by building a wheel directly from the source directory.  
So build backends ***may*** refuse to build an sdist.  
  
If the build backend is 100% sure it's never going to raise that exception, it ***may*** not define it.  
  
Build backends ***may*** implement an optional `get_requires_for_build_sdist` hook, `get_requires_for_build_sdist(config_settings=None)`, that allows to declare additional build requirements so that frontends will install them along those declared in `requires` when wanting to build an sdist.  
Backends that implement that ***must*** ensure that it returns a list of [PEP 508](https://peps.python.org/pep-0508/) dependency specifiers. Otherwise the default behavior is an empty list, `[]`.



##### Building a wheel



Similarly, build backends ***must*** implement a `build_wheel` hook, `build_wheel(wheel_directory, config_settings=None, metadata_directory=None)`, that ***must*** create a `.whl` file and ***must*** place it in the specified `wheel_directory`.  
  
Similarly, build backends ***must*** output the basename of the created wheel file as a Unicode string.  
  
Build backends ***must*** handle metadata consistently if they implement an optional hook called `prepare_metadata_for_build_wheel`, `prepare_metadata_for_build_wheel(metadata_directory, config_settings=None)`. Which means that if that hook was previously called, which will create a metadata directory etc., then the build backend must ensure that the final wheel's metadata matches the earlier metadata. **How will the build backend know about that?** The build frontend will communicate to them the information via the `metadata_directory`.  
Specifically, this hook ***must*** create a ***valid***, except for the `RECORD` file and signatures, `.dist-info` directory containing the wheel metadata inside the specified `metadata_directory`. So it'll be something like `{metadata_directory}/{package}-{version}.dist-info/`. This optional hook ***must*** return the basename of the `.dist-info` directory it creates, as a unicode string. I find this recurrent pattern really cool because that means that they really put time and effort into thinking about this interface and you can just expect things to work based on some few other things you already saw / know.  
Build backends ***may*** also create additional files inside that `metadata_directory` which will allow them to do stuff like record build-time decisions for re-use during the actual wheel build step etc.  
  
Since the hook is optional, build backends that don't implement it ***may*** ignore the `metadata_directory` in the `build_wheel` hook or raise an exception if it's not `None`.  
  
Similarly, build backends ***may*** implement an optional `get_requires_for_build_wheel` hook, `get_requires_for_build_wheel(config_settings=None)`, that allows to declare additional build requirements so that frontends will install them along those declared in requires when wanting to build a wheel.  
Backends that implement that ***must*** ensure that it returns a list of [PEP 508](https://peps.python.org/pep-0508/) dependency specifiers. Otherwise the default behavior is an empty list, [].  
  
So when it comes to wheels, there are two optional hooks.



##### Common between building an sdist and a wheel



Build backends ***must*** ensure they run with no issues in any isolated Python environment that installs the build requirements in `pyproject.toml` and those specified by the optional hooks `get_requires_for_build_*`. This means that they ***must*** ensure that both the mandatory hooks (`build_sdist` and `build_wheel`) and optional hooks (`get_requires_for_build_sdist`, `get_requires_for_build_wheel` and `prepare_metadata_for_build_wheel`) are able to be executed in that environment with no issues.  
Not only that, they ***must*** also ensure that any CLI tool provided by the build requirements is present in the environment's PATH. They also ***should*** ensure that the signatures of their hooks match both the order and the names of the arguments described in the PEP, whether they are positional or keyword arguments. This guarantees compatibility with the build frontend's expectations and prevents potential runtime errors due to mismatched interfaces.  
  
So build backends ***must not*** rely on any external package except those in the standard library and those defined in the build requirements.  
  
We're going to see it with the frontend's recommendations, but there is another layer of isolation applied to the build backend. Each of the hooks should be called in a subprocess. This allows build backends to change the global state of the subprocess without having an impact on the frontend's state. Any changes to environment variables are not leaked to the frontend's environment for example. This also means that build backends ***must*** ensure that they run the hooks with the working directory set to the root of the source tree, normally the build frontend does that but build backends must verify that as well. This ensures that all relative paths and file operations are correctly resolved within the context of the source tree.  
  
So the only interactions between the build frontend and build backend are **through the file system** and **the outputs of the hooks**.  
  
We see now that with this interface, there is a really good isolation between the build frontend and the build backend. The responsibilities are well assigned.  
  
One more thing, since build backends ***may*** store intermediate artifacts in cache locations or temporary directories, and since they mayhave to alter the working directory (if the `wheel_directory` is not specified for example). Build backends may fail when you don't have write access to these directories. The PEP says that build backends ***may*** fail when the source directory is read-only and if they can't build without creating or modifying any file in it, but it doesn't say whether they may or may not fail if they can't store the intermediate artifacts. But I guess in almost all situations they'll have the right to write to their cache locations or temporary directories.



##### Editable installs



We talked briefly about editable installs in the first part. They're installs that allow you to work on a project in a local source directory while ensuring that changes to the project's ***Python code*** are immediately reflected without the need for a new installation step. So if I install my toy package `my_package` as an editable install in another package and then change the code of `my_package` it should be reflected immediately in my other package.  
  
The focus is on the Python code because changes to build configuration files (like `pyproject.toml`), to entry points, or changes to non-Python source code (such as C extensions) may still require a new build and installation step to take effect.  
  
[PEP 660](https://peps.python.org/pep-0660/) follows closely [PEP 517](https://peps.python.org/pep-0517/) and adds three ***optional*** hooks.  
  
To maintain the build frontend - backend isolation, when you, in your `project2` ask from your build frontend to install a `package1` as an editable install, like `uv add --editable <path-to-package1>`, what happens is that the build frontend will ask from the build backend (of `package1`) to build an "editable" wheel, it goes through calling the `build_editable` hook. It's really like calling `build_wheel` with some minor differences that will let your build frontend provide `package1` as the editable install that you expect. We'll get into the details of that. This is important to understand, because the wheel here is a way to communicate between the build frontend and the build backend. Nothing else. It'll be discarded after the build frontend installs `package1` in your `project2`. (  
p`roject2` doesn't have to be structured as a package or anything by the way, it might not have a `build-system` table etc., the build frontend, in our case `uv` will case the build backend from `package1`.  
  
Let's get into the nitty gritty details!  
  
The first hook is, `build_editable`, `build_editable(wheel_directory, config_settings=None, metadata_directory=None)`. Build backends that implement that hook ***must*** ensure that it creates a `.whl` file that ***must*** be placed in the specified `wheel_directory`, and the hook ***must*** return the basename of the created wheel file as a Unicode string. It's similar to the other `build_*` hooks. Though as we said this wheel will be later on discarded, its filename must be [PEP 427](https://peps.python.org/pep-0427/) compliant. It doesn't need to use the same tags as build\_wheel but it must be tagged as compatible with the system. We'll get into that in the package formats section.  
  
Build backends ***may*** perform an in-place build of the distribution as a side effect of this hook, so that any compiled extensions or other built artifacts are ready to be used. Usually build backends place that in a `build` directory. Placing any compiled extensions (like C extensions) or built artifacts directly in the source directory allows them to be used directly when the package is imported.  
  
The generated wheel ***must*** comply with the wheel binary file format specification and ***must*** include a valid `.dist-info` directory. Metadata ***must*** be identical to that produced by `build_wheel` or `prepare_metadata_for_build_wheel`, except that the `Requires-Dist` field may include additional runtime dependencies necessary for the editable mechanism to function. What do we mean by that? The build time requirements or dependencies are those specified in `requires` in the `build-system` table of the `pyproject.toml` right. But since we want to install our package as editable, we might require some kind of mechanism (e.g. the [editables](https://pypi.org/project/editables/) package). Those dependencies are not build requirements, it's important to understand this in order to understand that the frontend - backend isolation is still maintained. Those are runtime dependencies that ensure that the mechanism editable works as intended. Build backends ***may*** add these run time dependencies in the `Requires-Dist` of the created wheel.  
  
Similarly, build backends ***may*** implement the `get_requires_for_build_editable`, `get_requires_for_build_editable(config_settings=None)` hook. It allows the declaration of additional build requirements needed for building an editable wheel. The hook ***must*** return a list of [PEP 508](https://peps.python.org/pep-0508/) dependency specifiers. If not defined, the default behavior is an empty list, `[]`.  
  
And similarly, build backends ***may*** implement the `prepare_metadata_for_build_editable`, `prepare_metadata_for_build_editable(metadata_directory, config_settings=None)` hook. This optional hook ***must*** create a valid `.dist-info` directory containing the wheel metadata inside the specified `metadata_directory`. The directory must comply with the wheel specification, except that it need not contain `RECORD` or signatures. The hook ***must*** return the basename of the created `.dist-info` directory as a Unicode string.  
And similarly, the build backend ***may*** also create additional files inside the `metadata_directory` to record build-time decisions for reuse during the actual wheel build step or for something else, and the build frontend must preserve, but otherwise ignore, such files.  
  
And the same again, if the build frontend previously called `prepare_metadata_for_build_editable` and requires the wheel to have matching metadata, it should pass the path to the `.dist-info` directory via the `metadata_directory` argument. The build backend ***must*** then ensure that the wheel's metadata matches the earlier metadata.  
  
There is a couple more things that are expected from the build backend, which are obvious. Build backends ***must*** ensure that the generated wheel, when installed, allows the package to be imported from its source directory. That's what we want right. And build backends ***may*** use different techniques to achieve this. I'll not get into the details of these techniques but we'll see how the first one works in an illustrative example. But you can read [this section from the PEP 660](https://peps.python.org/pep-0660/#what-to-put-in-the-wheel) to know more. The three examples that are provided are:  
- **Using `.pth` files**: This is what we're going to showcase with a toy example. It's just placing a `.pth` file in the wheel that adds the source directory to the Python path. This approach is simple but may include unintended files from the source directory. With this, `build_editable` won't add the extra dependencies that we talked about before in `Requires-Dist`.  
- **Proxy modules**: Create proxy modules that dynamically load code from the source directory. This method can provide a higher level of fidelity compared to path-based methods. More about that in the [editables](https://pypi.org/project/editables/) packages.  
- **Symbolic links**: Use symbolic links to mirror the source directory structure in the installed location.



##### Frontend Best Practices



We talked about the backend interface, but the PEP recommends quite a lot of things to build front ends as well. As expected there are no real recommendations when it comes to building sdists because they're quite direct, most recommendations are about building wheels.  
  
So build frontends ***may*** choose to call `build_sdist` first and then call `build_wheel` ***in*** the unpacked sdist when the user asks to build a wheel. This ensures consistency in building the wheels. So whether you build a wheel from the source tree or the source distribution you always get the same wheel for example.  
  
But we said that build backends might fail when it comes to building sdists, in that case the build frontend ***must*** fall back to calling `build_wheel` directly in the source directory which will ensure creating the wheel as asked by the user even if the sdist cannot be created.  
  
When it comes to building wheels, if the build frontend calls the `prepare_metadata_for_build_wheel` hook, it ***must*** pass the `metadata_directory` to the `build_wheel` hook of the build backend. We said that the build backend must ensure that the wheel metadata matches that metadata, so it has to know about the directory.  
If the build backend doesn't implement the `prepare_metadata_for_build_wheel` hook, then the build frontend should call `build_wheel` and get the metadata directly from the wheel.  
And obviously since that metadata impacts the build backend and the wheel building, the build frontend ***must*** preserve or ignore the `metadata_directory` and its files. For example, the build backend may store metadata that depends on build-time decisions, and it might need to reuse this information during the actual wheel-building step. Therefore, the frontend should not modify or interfere with the contents of the `metadata_directory`.  
  
Then there are all the best practices when it comes to setting up the environment where build requirements are to be installed and where the build backend will run.  
The most important thing here I think is that build frontends ***should*** set up an ***isolated*** Python environment. This isolation is key, it ensures that the build process is reproducible, it ensures that no external factors like the user's global Python environment will affect the build, no external module / library will be leaked into etc.  
For this goal, build frontends ***may*** use any mechanism to create this isolated environment. But just having an isolated Python environment is not enough obviously, we talked about it in the build backend interface but we should do it here as well, we need more isolation. And how to do that? build frontends ***should*** call each hook in a fresh subprocess. This allows build backends to change process-global state, such as environment variables or the working directory, without affecting the frontend or subsequent builds.  
But if that happens, we ***must*** ensure that any hook has access to the packages it needs. Which means that the hooks `get_requires_for_build_sdist` and `get_requires_for_build_wheel` ***must*** run into that isolated Python environment in order not to miss any package right, along the packages in `requires` in `pyproject.toml`. It also means that the `build_sdist` and `get_requires_for_build_sdist` ***must*** have access to the build requirements from `pyproject.toml` and `get_requires_for_build_sdist`. And similarly, the `build_wheel` and `prepare_metadata_for_build_wheel` ***must*** have access to the build requirements from `pyproject.toml` and `get_requires_for_build_wheel`. And obviously, any other new Python subprocesses spawned by the build environment ***must*** have access to ***all*** of the project's build requirements. This ensures that if the build backend spawns subprocesses, they will run in the same isolated environment and have access to the necessary packages.  
  
And this goes without saying, this isolated Python environment ***should*** be created freshly for every new build, not reused.  
  
There are also recommendations from the [PEP 660](https://peps.python.org/pep-0660/), when it comes to editable installs. They're almost the same as before but adapted to editable installs and with some minor differences.  
  
First, obviously, build frontends ***must*** install the "editable" wheels generated by the `build_editable` hook in the same way as regular wheels. This means using the standard wheel installation mechanism without any special treatment. After installing the editable wheel, build frontends ***must*** create a `direct_url.json` file in the `.dist-info` directory of the installed distribution, in compliance with [PEP 610](https://peps.python.org/pep-0610/). The url field ***must*** be a `file://` URL pointing to the project directory (the directory containing `pyproject.toml`), and the `dir_info` field must be `{"editable": true}`. This provides a standard way to record the source of the installed package and indicate that it is installed in editable mode.  
  
Build frontends ***must*** execute the `get_requires_for_build_editable` hook in an environment that includes the build requirements specified in `pyproject.toml`. This ensures that the hook has access to any necessary packages to determine additional build dependencies. Similarly, build frontends ***must*** execute the `prepare_metadata_for_build_editable` and `build_editable` hooks in an environment that includes both the build requirements from `pyproject.toml` and any additional requirements specified by the `get_requires_for_build_editable` hook. This ensures that the build backend has all necessary dependencies to build the editable wheel.  
if the build frontend previously called `prepare_metadata_for_build_editable` and requires the wheel to have matching metadata, it ***should*** pass the path to the `.dist-info` directory via the `metadata_directory` argument. And the build frontend ***must*** preserve, but otherwise ignore, such files. Finally, if the build frontend needs the metadata and the method is not defined, it should call `build_editable` and extract the metadata directly from the resulting wheel.   
  
The most important thing to understand about editable installs, is that the "editable" wheel is a fleeting object. It must be removed once the package is installed. As such, build frontends ***must*** not expose the wheel obtained from `build_editable` to end users. The wheel is intended as an ephemeral communication between the build backend and frontend and must be discarded after installation. It ***must not*** be cached or distributed.



##### Toy Example for Editable Installs



In this subsection we'll just have a quick editable install and see if everything is respected. We'll also **discuss a limitation of editable installs** as they're today and **one way to overcome it**.  
  
Let's create a package `my_package` with `uv init --package my_package`. We'll leave the `pyproject.toml` as it is, let's add a `my_module.py` in `src/my_package` and put the following code in it:



```python
def greet(name: str) -> str:
    return f"Hi there, {name}!"
```



Then put the following code in `__init__.py` for the CLI entry point:



```python
import argparse

from .my_module import greet


def main():
    parser = argparse.ArgumentParser(description="A simple CLI for greeting")
    parser.add_argument(
        "-n", "--name", help="Name to greet", type=str, default="world"
    )
    args = parser.parse_args()
    print(greet(args.name))
```



Now, we're going to create another project somewhere else, like in the same directory as where we created `my_package`, with `uv init my_project`. `cd` into it and do `uv add --editable ../my_package`.  
  
You should see something like:  
`Installed 1 package in 3ms  
my-package==0.1.0 (from file:///Users/username/path/to/my_package)`  
  
Let's verify the installation quickly, since we have an entry point, we should find the script in `my-project/.venv/bin` (if you're on Windows I think you should look into `Scripts` instead of `bin`). And we do find it, it's the file called `my-package`. That is **the entry point script**. We can do `.venv/bin/my-package -n Ed` and I get "Hi there, Ed!".  
  
You can also just do `uv run my-package -n Ed`.   
  
In case you're still skeptical that the installation worked, we can go create a Python file and import the package in it. But let's do something else. Let's see what `my-package` is exactly. Here's its content:



```python
#!/Users/username/path/to/my_project/.venv/bin/python
# -*- coding: utf-8 -*-
import re
import sys
from my_package import main
if __name__ == "__main__":
    sys.argv[0] = re.sub(r"(-script\.pyw|\.exe)?$", "", sys.argv[0])
    sys.exit(main())
```



The shebang line specifies the path to the Python interpreter to use when running this script. As you can see, it's the one from our venv. And then it imports the `main` function from `my_packag`e. If you forgot about entry points, [read this](https://reinforcedknowledge.com/a-comprehensive-guide-to-python-project-management-and-packaging-concepts-illustrated-with-uv-part-i/#What-are-entry-points) from my first article.  
  
So since we can run the command, it means that inside the venv, we're able to import `my_package`. All good!  
  
Now that we have verified that the install works. Let's go back to `my_package` and see if there is any wheel. As we said, the "editable" wheel is ephemeral. And indeed we don't find it. Since we're already here, let's change the "Hi there" in `greet` to "Hello". Let's go back to our project, `uv run my-package -n Ed`, and we get "Hello, Ed". Everything works perfectly.  
  
Let's continue our investigation.  
  
Let's go to `site-packages` and find what mechanism was used to enable this editable installation. So I do a `ls .venv/lib/python3.13/site-packages/` and what do I see? `_my_package.pth`. And what does it contain? The ***absolute path*** to `my-package`: `/Users/username/path/to/my_package/src`. This file tells Python to add `/path/to/my_package/src` to `sys.path`. In case you don't know, `sys.path` [is](https://docs.python.org/3/library/sys.html#sys.path) "A list of strings that specifies the search path for modules". That's where Python looks for when you import a module. It's an attribute of the `sys` module so if you want to know what is in the `sys.path`, you just have to `import sys` and `print(sys.path)`.  
  
The `.pth` is the **most important file** **in this case**. You can remove the `.dist-info` and you can still import and use your package. But if you use `uv`, since it's a tool, it'll **reinstall it as editable** `my-package`. If you remove the `.pth` file and import the package, you'll get  `ModuleNotFoundError: No module named 'my_package'`. And you won't be able to run the CLI obviously.  
  
Anyways, let's move on to see if the `.dist-info` respects what the PEP recommends. We do find the `direct_url.json` file that indicates that the package was installed in an editable mode (because you can add the `.pth` manually, it doesn't mean you have installed the package, you need the metadata).  
  
And inspecting its content, we do find that it's up to standard:  
`{"url":"file:///Users/username/path/to/my_package","dir_info":{"editable":true}}`  
With the `url` pointing to the location of `my_package` on the filesystem and `dir_info` indicating that the package is installed in editable mode `{"editable": true}`.  
  
So to sum it up:  
The build frontend asks the build backend to build a wheel, this wheel contains files (like `.pth` files or proxy modules) that, when installed, set up the package to import code directly from your source directory.  
When the build backend builds the wheel, the build frontend will install it. This installation process unpacks the wheel into `site-packages` (and `bin` in case you have entry points etc.), setting up the necessary links or proxy modules. The installation sets up `my_project` to import `my_package` directly from its source directory thanks to the `.pth`.  
The wheel file itself is discarded after installation.  
  
I hope this was a smooth intro for you in the world of installing wheels, and that it removes a bit of the magic. I went into all of these details to encourage you to inspect the files yourself. It's easy to do, you learn a lot by investigating this way, and don't be scared to break things.  
  
There's a limitation mentioned in the PEP but that we didn't address. Here's what the PEP [says](https://peps.python.org/pep-0660/#limitations):


> With regard to the wheel `.data` directory, this PEP focuses on making the `purelib` and `platlib` categories (installed into site-packages) “editable”. It does not make special provision for the other categories such as `headers`, `data` and `scripts`. Package authors are encouraged to use `console_scripts`, make their `scripts` tiny wrappers around library functionality, or manage these from the source checkout during development.



What are these `purelib`, `platlib`, `data`, `scripts` and `headers`? We'll get into it later on in the wheel format, but you can think of it as categories of files within a wheel, based on the type and intended installation location. `purelib` contains pure Python modules and packages (i.e., code that is platform-independent). These are installed into the `site-packages` directory. That's the case for our `my_package`. `platlib` contains platform-specific modules and packages, such as compiled extension modules (e.g., `.so` files). These are also installed into a `site-packages` directory.   
The other categories like `scripts` are the executable scripts to be installed into a directory like `bin` or `Scripts` (on Windows), `data` are data files intended for installation into a shared data directory, `headers` are C header files.  
All of these are like `dist-info` inside a wheel.   
  
During the installation of a wheel, files from the `purelib` and `platlib` categories are copied into `site-packages`, while the `scripts`, `data`, and `headers` are installed into their respective directories outside of `site-packages`.  
  
The PEP focuses only on making the `purelib` and `platlib` parts of a package editable. But, you have noticed that a change in our `main` function is reflected directly in the CLI in `my_project`, so how come? Well, we didn't do anything to have it since we used `uv init --package` but that's actually a suggestion from the PEP. Since executable scripts might not pick up changes in the source code unless they are reinstalled, the PEP suggests using `console_scripts` **entry points**, or generate small wrappers that invoke code from the package or manage these from the source checkout during development.   
  
**Having the entry point manages a lot of the headache, so it's a huge recommendation when it comes to making your scripts editable**.  
  
For the rest, unfortunately we can't do much for the moment. And to prove it to you, let's go to `my_package`, add a `data.txt` file, `echo "Ed" > src/my_package/data.txt`, let's add a `read_data` function that opens this file and prints its content. Go back to `my_project`, import the function. The import works because the package is installed as editable and this is `purelib`, but invoking the function won't work. You'll get a  `FileNotFoundError: [Errno 2] No such file or directory: 'data.txt'`. And this even if you reinstall the package!  
  
What you can do in this case is use `importlib.resources` to access the `data.txt` file and ensure they are found regardless of installation method.  
So you'll have something like `print(importlib.resources.read_text("my_package", "data.txt"))` and calling `read_data()` in `my_project` will work.



This completes our coverage of the editable installs. And we have also covered most of the interface of the build backends and build frontend. Let's tackle some minor stuff.



##### Customizing the Build Process



I don't think, from my experience at least, that many tools are totally compliant or in-tune together when it comes to implementing this part. Some do it partially, others do the same part but in a different way etc. So I'd suggest you look into the build systems you're using or search for the appropriate build systems if you have some needs that go beyond the basic features that the PEP offers, and which we'll discuss here.



###### Going beyond the PEP: Custom Hooks



The PEP doesn't talk about custom hooks, but many build systems allow you to add your own custom hooks that can be used to perform additional tasks at various stages of the build lifecycle.  
  
Just to illustrate what I said, let's take the editable installs toy example and let's add a custom hook. Since in my first article I had a comment from Abed asking about integrating [pyarmor](https://github.com/dashingsoft/pyarmor), let's do that! If you read this part Abed I hope it helps you and if you have any more questions don't hesitate to ask!  
  
So what [pyarmor](https://github.com/dashingsoft/pyarmor) does is, it allows us to obfuscate our code. Our goal here is to use a custom hook to run `pyarmor` on our code, to produce obfuscated code and that will be what goes into our wheel. The **sdist will still retain the source code**, because remember, the sdist helps others build your package for their platform, from scratch, so you can't provide them with just the obfuscated code. But **the sdist will also contain the obfuscated code**, and our **wheel building step will only get that obfuscated code from the sdist**.  
  
So with `pyarmor --help` and `pyarmor gen --help` I understand that I can do something like `pyarmor gen -O <output-dir> -r <source-dir>/<package-name>` and it'll obfuscate recursively the code in `<source-dir>/<package-name>` into `<output-dir>`. This should give you in `<output-dir>` a directory called `<package-name>` where there is your package obfuscated and a directory called `pyarmor_runtime_{some-number}`, for me the number was 000000, I don't know if that's always the case.  
  
So what we want to do with our hook is, when the building starts, **create a temporary directory** where **we'll copy our source code**, because we don't want to mess up anything right or overwrite our source code. And in that temporary directory we'll also **create another directory for the `pyarmor` output**.  
  
And **at the end of the build process, we'll delete that temporary directory**.  
  
If I were to do this in my source directory, I'd do something like:



```python
import shutil
import tempfile
from pathlib import Path


temp_dir = Path(tempfile.mkdtemp())
src_package = Path("src") / "my_package"
temp_package = temp_dir / "src" / "my_package"
temp_package.parent.mkdir(parents=True)
shutil.copytree(src_package, temp_package)

pyarmor_build = temp_dir / "pyarmor_build"
pyarmor_build.mkdir()

# Later on
shutil.rmtree(temp_dir)
```



This gives us a `temp_dir` that contains a `src/my_package` directory that contains our source code, and a directory `pyarmor_build` that will contain the output of `pyarmor`.  
  
Now let's run `pyarmor`. We'll do it this way:



```python
import subprocess


try:
    subprocess.run(
        ["pyarmor", "gen", "-O", str(pyarmor_build), "-r", str(temp_package)],
        check=True,
        capture_output=True,
        text=True,
    )
except subprocess.CalledProcessError as e:
    shutil.rmtree(temp_dir)
    raise RuntimeError(f"PyArmor failed: {e.stdout}\n{e.stderr}")
```



This should populate `temp_dir/pyarmor_build` with a directory `my_package` containing the obfuscated version of `src/my_package` and the `pyarmor_runtime_`.  
  
What we're going to do now is, replace our original source code that is in `temp_dir/src/my_package` with these. So what we want is to have `temp_dir/src/my_package` containing a directory `pyarmor_runtime_000000` (so that it gets included in the wheel later on) and the obfuscated code in `temp_dir/pyarmor_build/my_package`.  
  
There's also another thing we have to do, if you go check the obfuscated code, you'd see this `from pyarmor_runtime_000000 import pyarmor`, we'll have to replace it with `from .pyarmor_runtime_000000 import pyarmor`, so when the wheel is installed, in `site-packages` we have a structure like:  
`├── init.py  
├── my_module.py  
└── pyarmor_runtime_000000`  
  
So we must have a proper import.



```python
obfuscated_package = pyarmor_build / "try_pyarmor"
runtime_path = pyarmor_build / "pyarmor_runtime_000000"

final_package_dir = temp_dir / "src" / "try_pyarmor"
if final_package_dir.exists():
    shutil.rmtree(final_package_dir)

shutil.copytree(runtime_path, final_package_dir / "pyarmor_runtime_000000")

for file in obfuscated_package.iterdir():
    if file.suffix == ".py":
        dst_file = final_package_dir / file.name
        content = file.read_text()
        content = content.replace(
            "from pyarmor_runtime_000000 import __pyarmor__",
            "from .pyarmor_runtime_000000 import __pyarmor__",
        )
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        dst_file.write_text(content)
```



With that we have the core logic of our hook. Now let's implement it properly in a way that `hatchling` recognizes and in a way that it'll work as we intend it.  
  
First, since we're using `pyarmor` we have to install it. It's a build requirement. As we said before, the hooks (whether mandatory or optional) will be executed in an isolated Python environment that contains both the build requirements declared in `build-system` and those brought on by the `get_requires_*` hooks.  
  
Let's see what the `hatchling` documentation says about custom hooks, how to integrate them etc. There are two main pages to read: <https://hatch.pypa.io/dev/plugins/build-hook/reference/#hatchling.builders.hooks.plugin.interface.BuildHookInterface> and <https://hatch.pypa.io/dev/plugins/build-hook/custom/>.  
  
Since we just have one custom hook, let's keep things simple and put our code into a `hatch_build.py` at the root of your source directory. And let's add the `tool.hatch.build.hooks.custom` to our `pyproject.toml`  
  
Our `pyproject.toml` should look like this:



```
[project]
name = "my-package"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
authors = [
    { name = "ReinforcedKnowledge", email = "reinforced.knowledge@gmail.com" }
]
requires-python = ">=3.13"
dependencies = []

[project.scripts]
my-package = "my_package:main"

[build-system]
requires = ["hatchling", "pyarmor"]
build-backend = "hatchling.build"

[tool.hatch.build.hooks.custom]
```



And the source directory's structure is:  
`.  
├── README.md  
├── hatch_build.py  
├── pyproject.toml  
└── src  
└── my_package  
├── init.py  
└── my_module.py`  
  
Our custom hook should inherit from the `BuildHookInterface` as documented [here](https://hatch.pypa.io/dev/plugins/build-hook/reference/#hatchling.builders.hooks.plugin.interface.BuildHookInterface). So our `hatch_build.py` is going to look like this:



```python
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class PyarmorBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """
        This runs before the build process.
        We'll use this to run PyArmor and prepare the obfuscated files.
        """
        self.temp_dir = Path(tempfile.mkdtemp())
        
        src_package = Path(self.root) / "src" / "my_package"
        temp_package = self.temp_dir / "src" / "my_package"
        temp_package.parent.mkdir(parents=True)
        shutil.copytree(src_package, temp_package)
        
        pyarmor_build = self.temp_dir / "pyarmor_build"
        pyarmor_build.mkdir()
        
        try:
            subprocess.run(
                ["pyarmor", "gen", "-O", str(pyarmor_build), "-r", str(temp_package)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            shutil.rmtree(self.temp_dir)
            raise RuntimeError(f"PyArmor failed: {e.stdout}\n{e.stderr}")
        
        obfuscated_package = pyarmor_build / "my_package"
        runtime_dir = next(
            d.name for d in pyarmor_build.iterdir() 
            if d.name.startswith("pyarmor_runtime_")
        )
        runtime_path = pyarmor_build / runtime_dir
        
        final_package_dir = self.temp_dir / "src" / "my_package"
        if final_package_dir.exists():
            shutil.rmtree(final_package_dir)
            
        shutil.copytree(
            runtime_path,
            final_package_dir / runtime_dir
        )
        
        for file in obfuscated_package.iterdir():
            if file.suffix == '.py':
                dst_file = final_package_dir / file.name
                
                content = file.read_text()
                
                content = content.replace(
                    'from pyarmor_runtime_000000 import __pyarmor__',
                    'from .pyarmor_runtime_000000 import __pyarmor__'
                )
                
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                dst_file.write_text(content)

    def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str) -> None:
        """
        This runs after the build process.
        We'll use this to clean up our temporary files.
        """
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir)
```



**Remember**: you don't have to install `hatchling` or `pyarmor` in your project's virtual environment. They'll be installed in the build environment.  
  
You'd think that this is enough to do what we want. But go on, run `uv build`. It'll run successfully and you should see something like:  
`Building source distribution…  
Building wheel from source distribution…  
Successfully built dist/my_package-0.1.0.tar.gz and dist/my_package-0.1.0-py3-none-any.whl`  
  
But let's extract the wheel and see what's in there. You can do it this way from your source directory `unzip -d dist/extracted_wheel dist/my_package-0.1.0-py3-none-any.whl`.   
  
Let's see what we got in our wheel, `tree dist/extracted_wheel` gives:  
`├── my_package  
│   ├── init.py  
│   └── my_module.py  
└── my_package-0.1.0.dist-info  
├── METADATA  
├── RECORD  
├── WHEEL  
└── entry_points.txt`  
  
We lost the `pyarmor_runtime_000000` and on top of that if you open the `.py` files, they're not obfuscated at all!  
  
**Why is that?**  
Well the reason is pretty clear, we never told `hatchling` anything about our `final_package_dir`, how is it supposed to know that it's the one to be included in the wheel and not the original source code? There's no way. So how do we do that? Let's step back a little bit, we have our original source code, we have a `final_package_dir` that has the exact same structure as our original source code, but contains the obfuscated code right. And we want to include that instead of the original source code. There's a way to do that, and it is through the [build\_data](https://hatch.pypa.io/dev/plugins/build-hook/reference/#build-data) argument that's passed to `initialize`. As you can read in the documentation, "Build data is a simple mapping whose contents can influence the behavior of builds." and has the following field `force_include`, which we'll use, and the documentation says "This is a mapping of extra [forced inclusion paths](https://hatch.pypa.io/dev/config/build/#forced-inclusion), with this mapping taking precedence in case of conflicts".  
  
So we'll use it this way: `"file_path_to_include": "original_file_path"`, for example `"<temp-dir>/src/my_package/module.py": "my_package/module.py"`.  
  
Here's then our complete `hatch_build.py` (that you can download from <https://github.com/ReinforcedKnowledge/scripts-and-utils/blob/main/pyarmor_hatch_build.py>):



```python
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class PyarmorBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """
        This runs before the build process.
        We'll use this to run PyArmor and prepare the obfuscated files.
        """
        self.temp_dir = Path(tempfile.mkdtemp())
        
        src_package = Path(self.root) / "src" / "my_package"
        temp_package = self.temp_dir / "src" / "my_package"
        temp_package.parent.mkdir(parents=True)
        shutil.copytree(src_package, temp_package)
        
        pyarmor_build = self.temp_dir / "pyarmor_build"
        pyarmor_build.mkdir()
        
        try:
            subprocess.run(
                ["pyarmor", "gen", "-O", str(pyarmor_build), "-r", str(temp_package)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            shutil.rmtree(self.temp_dir)
            raise RuntimeError(f"PyArmor failed: {e.stdout}\n{e.stderr}")
        
        obfuscated_package = pyarmor_build / "my_package"
        runtime_dir = next(
            d.name for d in pyarmor_build.iterdir() 
            if d.name.startswith("pyarmor_runtime_")
        )
        runtime_path = pyarmor_build / runtime_dir
        
        final_package_dir = self.temp_dir / "src" / "my_package"
        if final_package_dir.exists():
            shutil.rmtree(final_package_dir)
            
        shutil.copytree(
            runtime_path,
            final_package_dir / runtime_dir
        )
        
        for file in obfuscated_package.iterdir():
            if file.suffix == '.py':
                dst_file = final_package_dir / file.name
                
                content = file.read_text()
                
                content = content.replace(
                    'from pyarmor_runtime_000000 import __pyarmor__',
                    'from .pyarmor_runtime_000000 import __pyarmor__'
                )
                
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                dst_file.write_text(content)
        
        build_data["force_include"].update({
            str(f): str(f).replace(f"{self.temp_dir}/src/", "")
            for f in self._get_all_files(final_package_dir)
        })

    def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str) -> None:
        """
        This runs after the build process.
        We'll use this to clean up our temporary files.
        """
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir)
    
    def _get_all_files(self, directory: str) -> list[str]:
        return [str(f) for f in directory.rglob('*') if f.is_file()]
```



Make sure to loop through the `final_package_dir` to not forget the `pyarmor_runtime` files!  
  
Now we're ready to go, let's do `uv build` again. Let's unzip the wheel and check the structure:  
`├── my_package  
│   ├── init.py  
│   ├── my_module.py  
│   └── pyarmor_runtime_000000  
│   ├── init.py  
│   └── pyarmor_runtime.so  
└── my_package-0.1.0.dist-info  
├── METADATA  
├── RECORD  
├── WHEEL  
└── entry_points.txt`  
  
And when you check the `.py` files you'll notice that they're obfuscated.  
  
We can go on and see that everything works well by creating a project and adding this wheel as its dependency:  
`uv init my_project  
cd my_project  
uv add ../my_package/dist/my_package-0.1.0-py3-none-any.whl`  
  
You should then get: `+ my-package==0.1.0 (from file:///Users/username/path/to/my_package/dist/my_package-0.1.0-py3-none-any.whl)`.  
  
And if you check `my_package` in `site-packages` you'll find the `pyarmor_runtime_00000` and **the obfuscated code**! Running `uv run my-package -n Ed` gives us "Hello, Ed!". So everything works!  
  
This is what you should distribute if you care about code obfuscation. I'm not expert in code obfuscation though so please take this with a grain of salt. The objective of this section is to demonstrate that tools can go beyond the standard and to demonstrate how to use custom hooks, specifically `hatchling` custom hooks.  
  
If you care about code obfuscation you wouldn't distribute the sdist, it contains the original source code.  
Also, in case you care (but we'll dive deeper into that later on), you'll notice in the sdist that it also contains `hatch_build.py`, we can remove that with a `hatch` `tool` table in the `pyproject.toml`. But I prefer to leave it here since you won't be distributing the sdist (in this case), it means that if you were to do that, you're distributing it to your colleagues or selling it to some company or something and in that case it'd be good to give them the custom hook as well for reproducibility of the wheel.  
Also notice that in the sdist there is a `.gitignore` that is exactly the same as the one in your source directory. It's for similar reasons.   
  
By the way, that one got nothing to do with the `.gitignore` with `*` that `hatch` adds in the `dist` directory to avoid committing by accident or something.  
  
That's it for our coverage of custom hooks! Let's move on for other ways of customizing the build process.



###### PEP 517: config\_settings



The `config_settings` is an argument **passed to all build backend hooks** and serves as an escape hatch for users to pass ad-hoc configuration into individual package builds. Build backends ***may*** interpret `config_settings` in any way they find suitable for their build process. This means that the keys and values in this dictionary can control various aspects of the build, such as compiler options, build variants, or any other backend-specific settings.  
  
Build frontends ***should*** provide mechanisms for users to specify arbitrary string-key/string-value pairs to populate the `config_settings` dictionary and ***may*** provide different mechanisms (not necessarily a CLI argument) for users to do so, such as configuration files or environment variables.  
  
In the case of `uv`, you can provide that using the `pyproject.toml` or `uv.toml` inside the `tool.uv` table.  
Example for `pyproject.toml`, from `uv`'s [reference](https://docs.astral.sh/uv/reference/settings/#config-settings):   
`[tool.uv]  
config-settings = { editable_mode = "compat" }`  
  
If a user provides duplicate keys, the frontend ***should*** combine the corresponding values into a list of strings. This ensures that all provided values are passed to the build backend for interpretation.



###### PEP 517: overriding build requirements



This, as the name suggests, is a way to override build requirements. It's quite essential even though we have allocated a small section for it. Taking the [example from the PEP](https://peps.python.org/pep-0517/#recommendations-for-build-frontends-non-normative), imagine you have a package that you want to build and that declares in its build requirements `foo >= 1.0`, but now there is version 1.1 of `foo`, if that version introduces a breaking change, you'd like to pin the version to `1.0`. That's the use case for options like `--build-with-system-site-packages`, which allows the build environment to include packages installed in the system's site-packages directory, or `--build-requirements-override=my-requirements.txt`, which allows using an alternate set of build requirements to be installed in the isolated build environment. Build frontends ***should*** provide such mechanisms that allow users to override the build requirements.  
  
In `uv` there isn't really a way to override the build requirements but `uv` offers the `--build-constraint` for the `uv build` command to constrain build dependencies given a `requirements.txt`-like file. So it only constrains the already specified build requirements to the versions in the `requirements.txt`-like file.



##### Backend Outputs



Build backends ***may*** print arbitrary informational text to `stdout` and `stderr`. Since build frontends ***may*** close `stdin` before invoking backend hooks to prevent unintended interactions or blocking behavior, build backends ***must not*** read from `stdin` when running their hooks.   
  
When the output streams are not connected to a terminal or console (i.e., when `sys.stdout.isatty()` returns `False`), build backends ***should*** ensure that any output they write to those streams is UTF-8 encoded. This is important because build frontends ***may*** be capturing the output programmatically, and using a consistent encoding like UTF-8 ensures that the output can be decoded and displayed correctly, regardless of the system's default encoding. But if captured output is not valid UTF-8, build frontends ***must not*** fail due to this. Build processes should be robust, and the inability to decode some output should not cause the entire build to fail. However, build frontends ***may*** choose not to preserve all of the information in that case.  
  
When the output stream is a terminal, build backends ***should*** present accurate information suitable for terminal display, like any program running in the terminal.



We have now covered everything that's related to build backends and build frontends interfaces and how they interact with each other. We'll now talk about the in-tree build backends that we mentioned in the beginning. I must say, this is a very particular use case and I have never done that before. I'll just present my understanding of this use case as I've read about it and as I've understood the PEP.



##### In-tree build backends



Two specific situations are mentioned for this use case of in-tree build backends:  
- **Self-hosting**: which are projects that use themselves to build themselves  
- **Project-specific backends**: these are projects that might have wrappers around existing backends and the wrapper is too project specific to distribute as a standalone and use as external build requirement. I guess this is the case of the following example of `backend-path`:  
`[build-system]  
# Defined by PEP 518:  
requires = ["flit"]  
# Defined by this PEP:  
build-backend = "local_backend"  
backend-path = ["backend"]`  
  
Here you have `flit` as backend, an external build requirement, but maybe you have some wrapper around it and **the code for the wrapper with respect to your source tree's root** is in "backend" and the path for the actual build backend is "local\_backend".  
  
The `build-system` table as defined in [PEP 518](https://peps.python.org/pep-0518/) and extended in [PEP 517](https://peps.python.org/pep-0517/) doesn't cover the in-tree build backends use case because `requires` takes in external build requirements and we have said before that the build frontend doesn't include the project's source tree in the build environment's `sys.path`. So there is no way to access this in-tree code.  
  
That's why PEP 517 came up with the `backend-path` key. This key **contains a list of directories, relative to the project root**, which the frontend will add to the start of `sys.path` when loading the build backend and running the backend hooks. This exposes the code in the directories listed in `backend-path` within the build environment. So your `build-backend` will have access to it.  
  
There are important restrictions associated with the `backend-path` key:  
- Directories in `backend-path` are interpreted as **relative to the project root** and ***must*** refer to locations within the source tree (after resolving symbolic links). This ensures that the project's source tree remains self-contained, avoiding dependencies on code outside the project's control. This makes sense right, otherwise your build is non-reproducible and might not even be well portable since if you want others (colleagues or contributors etc.) to build the package, they'll need this external source code as well. Frontends ***should*** check that the directories specified in `backend-path` are within the project root and ***fail*** with an error if they are not.  
- The backend code ***must*** be loaded from one of the directories specified in `backend-path`. This means it is not permitted to specify `backend-path` and then use a backend located elsewhere. This restriction ensures that backend-path is used solely for in-tree backends and not for configuring external backends. This also makes sense, otherwise the `backend-path` would be almost useless. Frontends ***may*** enforce that the backend code is loaded from one of the directories specified in `backend-path`.



This is it for our in-depth coverage of the build systems. Now we can go back to our higher level stuff, build a toy package and check on the `uv` command, and go in-depth again into distribution formats!



## The toy package



We'll be reusing the toy example from editable installs, but let's do this from scratch in case you didn't read that section. We'll also throw in a dependency, an optional dependency and a dependency group to see the differences later on when we investigate the installed package. We already know the answer from the [first part](https://reinforcedknowledge.com/a-comprehensive-guide-to-python-project-management-and-packaging-concepts-illustrated-with-uv-part-i/#Dependencies) of this article, the dependencies and optional dependencies go into the public package metadata but the dependency groups don't. But it's good to finally see that with your eyes.  
  
Let's initialize a folder as a package using `uv`, `uv init --package my_package`. If you don't know what the `--package` does, refer to the first part, or even better read the manual for it 😁.  
  
If you open the `pyproject.toml` you'll notice a `build-system` table which contains two keys, `requires` and `build-backend`. `uv` [doesn't have its own build backend yet](https://github.com/astral-sh/uv/issues/3957) (`uv` version 0.5.1 (f399a5271 2024-11-08)) so it defaults to [hatchling](https://pypi.org/project/hatchling/). If you want to configure some of `hatchling`'s options to tailor it to your project, you'll have to install [hatch](https://github.com/pypa/hatch) and use the `hatch build` command's options, or add the appropriate configuration manually in the `pyproject.toml`.   
`hatchling` is the build backend of `hatch`. There are other [build backends](https://packaging.python.org/en/latest/guides/tool-recommendations/#build-backends) that you can use.  
  
You'll also see a `[project.scripts]`that defines a CLI entry point in the `pyproject.toml`. If you're not going to provide one you can remove it, if you're going to provide one then edit it to reflect the correct function ([see](https://reinforcedknowledge.com/a-comprehensive-guide-to-python-project-management-and-packaging-concepts-illustrated-with-uv-part-i/#What-are-entry-points) for more information). I'll just leave it for the moment because I'm going to implement a simple CLI to check that everything went correctly later on when I'll install this toy package as a dependency for another project.  
  
**Note**: if you leave it and don't provide an implementation for the CLI, you won't encounter an error when building the package and you also won't encounter an error when installing it as a dependency for some other project.  
  
So let's do `uv add numpy pandas` and `uv add --optional excel openpyxl` and `uv add --dev ruff`. At this point you have a `pyproject.toml` that's similar to:



```
[project]
name = "my-package"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
authors = [
    { name = "ReinforcedKnowledge", email = "reinforced.knowledge@gmail.com" }
]
requires-python = ">=3.13"
dependencies = [
    "numpy>=2.1.3",
    "pandas>=2.2.3",
]

[project.scripts]
my-package = "my_package:main"

[project.optional-dependencies]
excel = [
    "openpyxl>=3.1.5",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[dependency-groups]
dev = [
    "ruff>=0.7.4",
]
```



I'll add a `my_module.py` in `src/my_package` and put the following code in it:



```
import numpy as np


def greet(name: str) -> str:
    return f"Hello, {name}!"


def array_size(arr: np.array):
    print(arr.size)
```



I'll also write the `main` function in `__init__.py` for my CLI:



```
import argparse

from .my_module import greet


def main():
    parser = argparse.ArgumentParser(description="A simple CLI for greeting")
    parser.add_argument(
        "-n", "--name", help="Name to greet", type=str, default="world"
    )
    args = parser.parse_args()
    print(greet(args.name))
```



The goal here is, after building the package, we'll use the build outputs (the source distribution and built distribution to be precise) and install them in different projects / virtual environments and try `my-package` or `my-package -n <some-argument>`, and also try the `array_size` function to verify that the `numpy` dependency has been correctly installed.



Now we can finally do the `uv build` and it'll build the package. It's as simple as that. We notice these logs:



```
Building source distribution...
Building wheel from source distribution...
Successfully built dist/my_package-0.1.0.tar.gz and dist/my_package-0.1.0-py3-none-any.whl
```



So we have built two objects, a **source distribution**, which is the `my_package-0.1.0.tar.gz` and a **wheel** which is `my_package-0.1.0-py3-none-any.whl`. Also notice how it says "building **wheel from source distribution**".



## Package formats






### High-level overview



**Related PEPs**: [517](https://peps.python.org/pep-0517/) for sdist, [518](https://peps.python.org/pep-0518/) for the `build-system` in pyproject.toml (covered in the first part), [625](https://peps.python.org/pep-0625/) for the sdist file name, [721](https://peps.python.org/pep-0721/) for source distribution archive features, [427](https://peps.python.org/pep-0427/) for the wheel format, [425](https://peps.python.org/pep-0425/), [600](https://peps.python.org/pep-0600/) and [656](https://peps.python.org/pep-0656/) for the naming and tagging of the `.whl` file,  
  
**Specifications**: [Core metadata specifications](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata) for `PKG-INFO`, [pyproject.toml specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/#pyproject-toml-spec) for `pyproject.toml`, [binary distribution format](https://packaging.python.org/en/latest/specifications/binary-distribution-format/#binary-distribution-format) specification, [platform compatibility tags](https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/#platform-compatibility-tags) (for the `.whl` files).



#### Source distribution



A **source distribution**, or **sdist**, is a standardized compressed archive format `.tar.gz` that **contains the source code of the package**, **a metadata file** named `PKG-INFO` that follows the [core metadata specifications](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata) and that enables packaging tools to efficiently access metadata without needing to recompute it, a **build configuration** such as the `pyproject.toml` which defines the build system requirements and configurations.   
  
Sdist can also contain other files in your source tree. Some tools allow you to exclude or include some files depending on how they operate. And build systems can store whatever else information they need in the sdist to build the project.  
  
Most tools include the `.gitignore` from your source tree.  
  
The archive follows a standardized naming pattern: `{package_name}-{version}.tar.gz` as specified in [625](https://peps.python.org/pep-0625/).  
  
In our example, the file name is `my_package-0.1.0.tar.gz`.  
  
Our sdist looks like this:  
`├── .gitignore  
├── .python-version  
├── PKG-INFO  
├── README.md  
├── pyproject.toml  
├── src  
│   └── my_package  
│   ├── init.py  
│   └── my_module.py  
└── uv.lock`  
  
As you see, we have the `.gitignore`, the `.python-version`, the `uv.lock` and the `README.md`. These are not required. But our build system (`uv` as build frontend and `hatchling` as build backend) includes them. The `PKG-INFO` follows its specification well:  
`Metadata-Version: 2.3  
Name: my-package  
Version: 0.1.0  
Summary: Add your description here  
Author-email: ReinforcedKnowledge reinforced.knowledge@gmail.com  
Requires-Python: >=3.13  
Requires-Dist: numpy>=2.1.3  
Requires-Dist: pandas>=2.2.3  
Provides-Extra: excel  
Requires-Dist: openpyxl>=3.1.5; extra == 'excel'  
Description-Content-Type: text/markdown  
  
# My Package  
  
A simple Python package to demonstrate building and publishing.`  
   
And we have the source code as is in our sdist.



#### Wheel



A **wheel** is a **binary distribution format** for Python packages. Though this might sound complex, it's just a `ZIP`-format archive with a specially formatted file name and the `.whl` extension. What is the difference with an sdist? A wheel is pre-built and all the necessary files it contains are organized precisely as they need to be within the user's Python environment. The wheel format is designed so that the installation process involves simply unpacking the wheel and placing its contents into the appropriate directories without any additional build steps or processing. It **contains the source files** (`.py`), **any compiled extensions** (such as shared object files `.so`), and **metadata directory** `.dist-info` that contains at least the key files `METADATA` for the package metadata, similar to `PKG-INFO` in sdists, `RECORD` which is a list of all files included in the wheel along with their hashes, which allows for integrity verification during installation, and `WHEEL` which specifies the wheel specification version and other *build-related* metadata.  
  
There's also what's called a universal wheel, which is for projects that support both Python 2 and Python 3 and don't contain C extensions.  
  
The wheel file name follows the rule `{distribution}-{version}(-{build tag})?-{python tag}-{abi tag}-{platform tag}.whl`.   
  
In our example, the file name is `my_package-0.1.0-py3-none-any.whl` which means that there is no build tag, that it's a Python 3 wheel, thus the `py3`, a generic Python version tag, no ABI tag because our package is a pure Python package, thus the `none`, and also since it's a pure Python package, our wheel is valid for any platform, thus the `any`.  
  
Let's see what this ZIP archive contains! Its structure is:  
`├── my_package  
│   ├── init.py  
│   └── my_module.py  
└── my_package-0.1.0.dist-info  
├── METADATA  
├── RECORD  
├── WHEEL  
└── entry_points.txt`  
  
So we do find our source code, and we do find the required files `METADATA`, `RECORD` and `WHEEL` in `.dist-info`. We also notice the `entry_points.txt` in `.dist-info` which follows the [entry point specification](https://packaging.python.org/en/latest/specifications/entry-points/) and indicates our entry point:  
`[console_scripts]  
my-package = my_package:main`  
  
We can check that the `METADATA` contains all relevant dependencies and doesn't contain the group dependencies:  
`Metadata-Version: 2.3  
Name: my-package  
Version: 0.1.0  
Summary: Add your description here  
Author-email: ReinforcedKnowledge reinforced.knowledge@gmail.com  
Requires-Python: >=3.13  
Requires-Dist: numpy>=2.1.3  
Requires-Dist: pandas>=2.2.3  
Provides-Extra: excel  
Requires-Dist: openpyxl>=3.1.5; extra == 'excel'  
Description-Content-Type: text/markdown  
  
# My Package  
  
A simple Python package to demonstrate building and publishing.`  
  
So we have the `Requires-Dist` for our project dependencies, we have the `Provides-Extra` for the optional dependency. And no sign of group dependencies as it should be!  
  
Our `RECORD` file is as expected, contains the hash for each file in our wheel:  
`my_package/init.py,sha256=lQSoRh2LDY_YnJ2fSbC6Xy01W98G5BxJXkBXw2_hrrI,300  
my_package/my_module.py,sha256=9Q-blvjsS3vuu5XXY1VRiwYSlnTHcheoa9_S23ia75c,131  
my_package-0.1.0.dist-info/METADATA,sha256=z5--7aSka3XZCwwQrv8WWrWoj6KB1t-C_Q0nfWydaDE,427  
my_package-0.1.0.dist-info/WHEEL,sha256=C2FUgwZgiLbznR-k0b_5k3Ai_1aASOXDss3lzCUsUug,87  
my_package-0.1.0.dist-info/entry_points.txt,sha256=st7cqbObeIT1rDRatKegLr-KJ4RsN0_Curl0zbdJcec,47  
my_package-0.1.0.dist-info/RECORD,,`  
  
And finally the `WHEEL` file indicates that our package is `purelib`, which is the case, and follows the `1.0` wheel specification, and the tool that built the wheel was `hatchling`:  
`Wheel-Version: 1.0  
Generator: hatchling 1.26.3  
Root-Is-Purelib: true  
Tag: py3-none-any`  
  
When installing the wheel in a project, it'll extract the source code into some `site-packages` (see the details for the difference between `Root-Is-Purelib` is true or false) in your project's Python environment. And it'll place alongside it the `.dist-info` package.  
  
Our build system as it is doesn't compile the `.py` into `.pyc`, but when we'll run some code, the code and the dependencies it relies on to run will be compiled accordingly.  
  
If you have other files in your wheel that might go into the `.data` directory of a wheel, these will be placed in the correct places as well by the installer (see the deeper dive for more details).  
  
Notice how the wheel doesn't contain the `pyproject.toml` as opposed to the sdist!



### Details



#### Source distribution



We mentioned all the important things to know about sdists in the high-level overview. I'll go into some minor details here and what is in the specification.



##### Minor details



The source distribution is built from the **source tree**, which is the source code, and the [PEP 517](https://peps.python.org/pep-0517/) and [518](https://peps.python.org/pep-0518/) compliant `pyproject.toml`. It's similar to a VCS checkout.  
  
Though we say that "we build the source distribution", it's an **"unbuilt" format**. It's just a compressed archive containing the raw source code, some metadata and build configuration. It's not in a ready to install state and installers like `pip` generally build wheels from them when no wheel is found.  
  
This makes **sdists inappropriate for packages that contain non-Python code** that requires compilation like C extensions. It's not an issue in and of itself, but the end user will need to have the necessary build tools to compile the extensions or integrate the non-Python code when building your package. This doesn't mean that you can't use libraries that rely on external (to Python) libraries like Numpy, because when the user will install your package (from source) with a tool like `pip`, the tool will look for the appropriate binary distribution for the declared dependencies for the user's platform.  
  
On the other hand, **sdists offer portability and flexibility**. You can give your sdist to anyone, and with the proper tools it can build your package for its platform. It's often what some package managers, such as the ones for Linux distributions, will do. Use the source distribution as the primary source for building packages in their respective ecosystems. Which makes sense because they have the tools to build your package appropriately for their ecosystem. It's also a good fallback for when tools don't find the appropriate wheel for the end user's environment, they can try to build it from the sdist. And in the worst case, sdists enable end users to easily make the necessary modifications to adapt the package to their environment.  
  
You can inspect a source distribution using the Python standard library [tarfile](https://docs.python.org/3/library/tarfile.html), or use the `tar` command on Unix-like systems, e.g. `tar -xvf`.  
  
It's **recommended and common practice to distribute both the source distribution and the wheels**.  
  
**Notes**:  
- You don't need a `pyproject.toml` to build a source distribution, but that's the standard way of doing it. There is a legacy format for sdists that I won't delve into.  
- The tools include `.gitignore` in the sdist in general because that helps ensure consistency between the sdist and the wheel built from it. Otherwise, when building the wheel we might, by accident, include some files that should not be included.  
- You don't need a source distribution to build a wheel. You can build a wheel directly (we have talked about that in the build systems' interfaces).



For the sake of completeness, I'll just restructure the recommendations from the specification in a way that allows me to retain that information. Most of it will be copy pasted from the specification, but I'll explain the non directly understandable stuff like the recommendations with respect to unpacking files or links, and permissions.



##### Specification: Sdist Format



- Compression format:
  - **Tar format only**: Zip archives or other compression formats are **not allowed** *at present*. Tar archives ***must*** be created in the modern POSIX.1-2001 pax tar format, which uses UTF-8 for file names.
- Filename:
  - The file name of an sdist ***must*** be in the form `{name}-{version}.tar.gz`, where `{name}` is normalized according to [name normalization specification](https://packaging.python.org/en/latest/specifications/name-normalization/#name-normalization), and `{version}` is the canonicalized form of the project version, see [version specifiers](https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers).
  - The name and version components of the filename ***must*** match the values stored in the metadata contained in the file.
  - Code that produces a source distribution file ***must*** give the file a name that matches this specification. This includes the `build_sdist` hook of a build backend (see [build backend interface for building a source distribution](#Building-a-source-distribution)).
  - Code that processes source distribution files ***may*** recognize source distribution files by the `.tar.gz` suffix and the presence of precisely one hyphen in the filename. Such code ***may*** use the distribution name and version from the filename without further verification.
- Archive content:
  - Top-level directory:
    - An sdist contains a ***single*** top-level directory named `{name}-{version}`, which contains the source files of the package.
    - The name and version ***must*** match the metadata stored in the file.
  - Required files: the directory ***must*** contain
    - a `pyproject.toml` file as specified in the pyproject.toml [specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/#pyproject-toml-spec).
    - a `PKG-INFO` file containing metadata as described in the [core metadata specification](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata). The metadata ***must*** conform to at least version 2.2 of the metadata specification.
  - The source tree contained in an sdist is ***expected to*** include the pyproject.toml file.
  - Build systems ***may*** add any other files required for the build or they see fit.



##### Specification: Extraction of Sdist



- General extraction guidelines:
  - When extracting a source distribution, tools **must** either use `tarfile.data_filter()` (e.g., `TarFile.extractall(…, filter='data')`), or adhere to the specific guidelines detailed in the paragraphs below about files, links, permissions and mode below.
  - On Python interpreters without `hasattr(tarfile, 'data_filter')` ([PEP 706](https://peps.python.org/pep-0706/)), tools that normally use that filter (directly or indirectly) ***may*** warn the user and ignore this specification. The trade-off between usability (e.g., fully trusting the archive) and security (e.g., refusing to unpack) is left up to the tool in this case.
  - Tools that do not use the `data_filter` directly (e.g., for backward compatibility, allowing additional features, or not using Python) ***must*** follow the specific guidelines detailed in the paragraphs below about files, links, permissions and mode below.
- Guidelines when it comes to files:
  - Invalid files: tool ***must not*** unpack the following files, they ***should*** notify the user if they encountered such files, and they ***may*** abort with a failure.
    - **Files that**, when extracted, **would be placed outside the intended destination directory**. This can happen if file paths are absolute (starting with a `/`) or if they include path traversal elements like `../` that move up the directory tree.
    - **Device files** (Including pipes), which are special files that represent hardware devices or inter-process communication endpoints. These are not regular files and can pose security risks. Example: `/dev/null`.
  - Leading slashes in filenames ***must*** be dropped. This prevents files from being extracted to the root directory or other unintended locations on the file system.
  - Files with `..` components: tools ***may*** handle them as invalid files, but are ***not required*** to do so. These `..` elements can navigate up the directory structure, potentially extracting files outside the intended destination directory.
- Guidelines when it comes to files:
  - Invalid links: tool ***must not*** unpack the following links, they ***should*** notify the user if they encountered such files, and they ***may*** abort with a failure.
    - Links that point outside the destination directory. When we talk about links, we include both **symbolic links** (symlinks) which are files that reference or act as pointers to other files or directories, and **hard links** which reference the data (instead of another file) that is included in another file.
  - Links that point to files not included in the archive: tools ***may*** handle them as invalid files, but are ***not required*** to do so.
  - Tools ***may*** choose to extract links as regular files instead of actual links. This means that they'll use the content from the linked file included in the archive and create a standard file, ignoring the link. This approach protects against some of the potential security risks associated with links.
- Permissions:
  - For each file or directory, tools ***must*** handle permissions in one of the following ways:
    - Use default permissions: apply the system's standard permissions for newly created files and directories, ignoring the permissions specified in the archive.
    - Set according to archive: Use the permissions exactly as they are specified in the archive.
    - Apply standard permissions: Set permissions to a common default:
      - Non-executable files: `rw-r--r--` (`0o644`), meaning the owner can read and write the file, while others can only read it.
      - Executable files and directories: `rwxr-xr-x` (`0o755`), allowing the owner full access, and others to read and execute but not write.
  - Executable bit ***recommendation***: preserve the *executable* permission for files that are intended to be run as programs or scripts. This helps maintain the functionality of executable files after extraction. In this context, bits or mode bits are what determine the permissions of a file or directory, specifying who can read, write, or execute it. It's the characters in `rw-r--r--`. So the recommendation says that if tools decide to change these bits, it's recommended they keep the executable bit at least.
  - High mode bits recommendations: ***must*** be cleared. High mode bits are special permission bits. For example the `setuid` high mode bit is the 4th bit in the octal representation and when it is set on an executable file, it allows the program to run with the permissions of the file owner, rather than the user running it. Example: `-rwsr-xr-x`



##### Other notes



- Tool authors are encouraged to consider how hints for further verification in `tarfile` documentation apply to their tools. This includes validating file paths, handling special file types appropriately, and ensuring that the extraction process does not introduce security vulnerabilities.
- At the time of this writing, the `data_filter` follows the guidelines specified above, but it may change in the future.



This concludes our coverage of sdists. Let's get to wheels!



#### Wheel



We said all the important thing to know about wheels in the high-level overview as well. I'll go into some minor details here and what is in the specification. And again, I'll copy paste a lot of the specification for the sake of completeness, I'll just explain what required from my a thought or two.



##### Minor Details



Wheels do not include compiled Python bytecode (.pyc files). Installers generate these during installation, since it's not time consuming and that way we can ensure they are optimized for that particular environment.  
  
Wheels can be platform-specific due to compiled extensions or platform-dependent code, but pure-Python wheels are generally platform-independent and can be used across different systems. Thus the `any` in the filename of our `.whl`.  
  
You can build wheels directly from the source code without first creating a source distribution.  
  
Since wheels are ZIP archives, you can inspect their contents using the built-in Python libarry [zipfile](https://docs.python.org/3/library/zipfile.html) or the CLI utility `unzip`.  
  
While it's technically possible to import Python code directly from a wheel file placed on `sys.path`, it is discouraged and many packages assume they have been fully installed and may not function correctly when run from a zip archive. I think it's the case for most linters and static analysis tools which require the packages to be installed to do a proper job.  
  
Tools might add additional files to the `.dist-info` on top of what's required by the specification. For example with our toy package, we have the `entry_points.txt` containing entry points as specified in the [entry points specification](https://packaging.python.org/en/latest/specifications/entry-points/).



##### Specification: Wheel format



- Compression format:
  - **ZIP format**: ZIP-format archives with a `.whl` extension.
- Filename:
  - ***Must*** follow the pattern: `{distribution}-{version}(-{build tag})?-{python tag}-{abi tag}-{platform tag}.whl`
    - distribution: the normalized name of the package.
    - version: the version of the package.
    - build tag (optional): a build number used to differentiate multiple builds of the same version. It must start with a digit and acts as a tie-breaker if two wheel filenames are identical in all other respects. Build numbers are typically used when you need to rebuild a binary distribution without changing the package's version number. This can happen when the build environment changes or when building a pre-release distribution. Build numbers are not considered part of the package's version as defined in [PEP 440](https://peps.python.org/pep-0440/). This means tools and systems that rely on version numbers may not recognize changes in build numbers. So, do not use build numbers when creating new distributions that need to be referenced externally, instead increase the package's version number like from 0.1.0 to 0.1.1.
    - python tag: the Python implementation and version tag for which the wheel is compatible with, e.g., py3.
    - abi tag: specifies the ABI (Application Binary Interface) compatibility, e.g., cp37m, abi3, none.
    - platform tag: indicates the platform the wheel is compatible with, e.g., linux\_x86\_64, any.
  - Distribution names ***should*** be normalized according to [name normalization](https://packaging.python.org/en/latest/specifications/name-normalization/#name-normalization). Tools consuming wheels ***must*** accept names of an earlier specification.
  - Versions ***should*** be normalized as well, according to [version specification](https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers).
  - Tools producing wheels ***must*** verify that filenames do not contain hyphens (`-`) to prevent ambiguity in parsing.
  - Wheel filenames are Unicode. Inside the archive, filenames are UTF-8 encoded.
- Archive content:
  - Root directory:
    - Contains all files to be installed into `purelib` or `platlib`, as specified in the `WHEEL` metadata file. These directories typically map to `site-packages`.
  - `.dist-info` directory:
    - Contains metadata about the package and the wheel itself. It is named `{distribution}-{version}.dist-info`.
    - Required files:
      - `METADATA`: contains package metadata, similar to `PKG-INFO` in sdists. Must be in Metadata version 1.1 or greater format.
      - `WHEEL`: metadata specific to the wheel in a simple key-value format. The keys are:
        - `Wheel-Version`: the version of the wheel specification used.
        - `Generator`: the build backend used to create the wheel, along with its version.
        - Root-Is-Purelib: whether the root of the archive should be installed into `purelib` (value is then `true`) or `platlib` (value is then `false`).
        - `Tag`: the wheel's compatibility tags, corresponding to the python tag, abi tag, and platform tag in the filename. You can have many values for this same key, each key-value pair on a different line.
        - `Build`: the build number, if specified.
      - `RECORD`: a list of all files in the wheel (except `RECORD` itself) along with their hashes and sizes. It also doesn't contain `RECORD.jws` and `RECORD.p7s`. Each line includes the file path, a hash of the file contents, and the file size. The hash algorithm ***must*** be SHA-256 or better.
    - Archivers are encouraged to place the `.dist-info` files physically at the end of the archive. This enables some potentially interesting ZIP tricks including the ability to amend the metadata without rewriting the entire archive.
    - Installers ***should*** warn if `Wheel-Version` is greater than the version they support.
    - Installers ***must*** fail if Wheel-Version has a greater major version than they support.
  - `.data` directory:
    - Contains subdirectories with files that are not installed inside `site-packages`. This includes:
      - Scripts: executable scripts to be installed into the user's `scripts` or `bin` directory.
      - Headers: Header files for C extensions.
      - Documentation: User manuals or other documentation files.
      - Data: Additional data files required by the package.
  - Wheels do not contain, in general, `.pyc` files.
  - Wheels do not contain `setup.py` or `setup.cfg`.



##### Specification: Wheel Installation



These are the steps that installers take.



**Unpacking**:



- Parse the `WHEEL` file:
- Compatibility check:
  - Check that the `Wheel-Version` specified in the `WHEEL` file is supported.
  - Warn if the minor version is greater than what the installer supports.
  - Abort if the major version is greater than what the installer supports.
- Determine Installation Root
  - If `Root-Is-Purelib` is `true`, unpack the archive into `purelib` (typically `site-packages`).
  - If `Root-Is-Purelib` is `false`, unpack the archive into `platlib` (generally it's `site-packages` as well but we'll talk about that later on).



What we mean by unpacking here is literally unpacking the ZIP archive into either `purelib` or `platlib`. This is not only for the package's code, but also `.dist-info` which is placed alongside it. Many dependency managers and other tools rely on `.dist-info` to manage dependencies and installed packages.



**Spreading**:



- Handle the `.data` directory:
  - Move each subdirectory's contents to their respective destinations as defined by the installation scheme.
    - Each subdirectory within `.data` corresponds to a key in the dictionary of destination directories defined by `sysconfig`'s [installation paths](https://docs.python.org/3/library/sysconfig.html#installation-paths) (e.g., `scripts`, `headers`, `data` and even `purelib` and `platlib`).
- Script processing:
  - Update scripts in the `scripts` directory that start with `#!python` to point to the correct interpreter (the interpreter of the environment where the package is installed, as we say in the [toy example for editable installs](#Toy-Example-for-Editable-Installs)).
  - On Unix systems, installers may need to add the executable bit to these scripts if the archive was created on Windows.
- Update `RECORD`:
  - Update the `RECORD` file to reflect the installed paths, it has to list all installed files along with their hashes and sizes.
- Clean up:
  - Remove the now-empty `.data` directory.
  - Compile any installed `.py` files to `.pyc` files. Uninstallers should be capable of removing these compiled files during uninstallation, even if they are not listed in `RECORD`. This behavior depends a lot on the tools used. That's why I haven't put it in a separate place. Installing with `uv` doesn't compile `.py` files. But when you're going to use something from a package, then it's going to compile the python files of that package and the dependencies needed to run what you want to run into `.pyc` (you're going to find that in `__pycache__`).



Recommendations for installers:



- ***Must*** verify all file hashes in `RECORD` against the extracted file contents, except for `RECORD` itself obviously.
- ***Must*** fail installation if any file is missing or has an incorrect hash.



##### Purelib and Platlib



`purelib` stands for pure Python code while `platlib` stands for platform-specific code.  
  
The root of a package is either `purelib` or `platlib`, can't be both, but the package itself can be a mix. In that case (mix), the root contains both folders.  
  
Why the distinction? `purelib` packages are installed in `site-packages`, while that might not be the case for `platlib`. It depends on the platform. They're also installed in `site-packages` but not the same path inside the `venv`. The [example from the specification](https://packaging.python.org/en/latest/specifications/binary-distribution-format/#what-s-the-deal-with-purelib-vs-platlib): "Fedora installs pure Python packages to ‘/usr/lib/pythonX.Y/site-packages’ and platform dependent packages to ‘/usr/lib64/pythonX.Y/site-packages’". Notice the `lib` vs `lib64`.  
  
Since a package can contain both `purelib` and `platlib`, there is a rare case where `Root-Is-Purelib` is false but also have some files in `purelib`. Now, if you have two versions of your wheel , one that is pure Python and works on any platform, and another one that is platform specific, and if you have duplicate code between the two. Example: you have "low-level" code written in Python and is in the pure Python wheel and you have "low-level" code written in C or something and is in the platform-specific wheel. Both wheels share the same pure Python code that is the higher level API. If in the platform-specific wheel, all of your code is in `platlib`, and you install both wheels on a platform that separates `purelib` and `platlib` during installation, then you might have some issues like code going out of sync etc. One way to solve that is to pure the non-platform dependent code in the platform-dependent wheel in the `purelib` folder, so that when installing both wheels, all the non-platform dependent code will go into the same place.



##### Wheels and signatures



[From the specification](https://packaging.python.org/en/latest/specifications/binary-distribution-format/#why-does-wheel-include-attached-signatures): "Attached signatures are more convenient than detached signatures because they travel with the archive. Since only the individual files are signed, the archive can be recompressed without invalidating the signature or individual files can be verified without having to download the whole archive".  
  
There are three ways to sign a wheel.   
  
- The mandatory `RECORD` file which lists all files (except for `RECORD` and the optional `RECORD.jws` and `RECORD.p7s`) in the wheel along with their hashes and sizes. Its entries are in the format `filepath, digestname=urlsafe_b64encode_nopad(digest), file_size`, where `digestname` is the hash algorithm's name, and `rlsafe_b64encode_nopad(digest)` provides a base64 URL-safe encoding of the digest without padding (`=` characters).  
  
- Optional: `RECORD.jws`. Contains one or more signatures in the JSON Web Signature JSON Serialization (JWS-JS) format. It is adjacent to `RECORD`. You can use it to sign `RECORD` itself, which is cool. The `RECORD` file is hashed using the **same algorithm** as the one used in `RECORD`, and the hash is then included as the payload in the JWS (see [example](https://packaging.python.org/en/latest/specifications/binary-distribution-format/#signed-wheel-files)).   
  
- Optional: `RECORD.p7s`. ***Must*** contain a detached S/MIME format signature of `RECORD`. It's an alternative to JWS that allows users to use existing public key infrastructure with wheel files.  
  
`RECORD.jws` and `RECORD.p7s` can only be added after `RECORD` is generated.  
  
Installers are not required to understand or verify digital signatures, but they ***must*** verify the hashes in `RECORD` against the actual files extracted. If any file's hash does not match or if any hash is absent, the installation ***must*** fail. This ensures the wheel has not been tampered with.  
A separate signature checker can then be used to verify the signature in `RECORD.jws` or `RECORD.p7s`, to check that the signature is valid and that the hash of `RECORD` matches the one in the signature. There is nothing said about whether to fail the installation or not if the signature is not valid in this case. But I guess the appropriate course of action is to fail the installation as well.



That wraps it up for wheels! Let's go back to our toy package and to `uv`!



## Some uv build options



As we said in "[overriding build requirements](#PEP-517-overriding-build-requirements)" in the “customizing the build process" section, `uv` doesn't provide a way to completely override the build requirements, but it's such a rare case that I guess you'll almost never use it. The [example mentioned in PEP 517](https://peps.python.org/pep-0517/#build-environment) itself speaks about using different versions for the build requirements, and that's what `uv` offers through the `--build-constraint` option that takes in a `requirements.txt`-like file that contains pinned versions and `uv` uses it to override those specified in build requirements. So neither does the `requirements.txt` file replace all the build requirements with the requirements specified in it, nor does it add new packages.  
  
There are options to verify the hashes of the requirements in `requirements.txt` that we feed to `--build_constraint`. The `--require-hashes` checks if ***all*** the packages have hashes. The requirements' versions must be pinned to exact versions or specified via URL. It doesn't work on Git dependencies, on editable dependencies and on local dependencies if they're not pointed to a wheel.  
The other option is `--verify-hashes`, validates the ***provided*** hashes. So if you want to make sure that all the packages in your `requirements.txt` have valid hashes you must use both `--require-hashes` and `--verify-hashes`.  
  
The `uv build` command takes a `[SRC]` argument which is the current working directory by default.  
The rest of the options are easy to understand, like `--package` to build a specific package in the workspace, `--all-packages` to build all packages in a workspace, `-o` or `--out-dir` to specify the output directory where the built distributions will be saved; defaults to the `dist` subdirectory, `--sdist` to only create the sdist, `--wheel` to only create the wheel.



# uv publish



Now that we have both our sdist and our wheel. We can publis them!



Publishing is relatively easy I believe. You publish your packages to an index.  
  
An index is just a storage location for packages, like GitHub for `git` repositories, or Docker Hub for Docker images.. [PyPI](https://pypi.org/) (Python Package Index) is the most used index and it's a public one. Organizations or individuals can also set up private indexes for internal use.  
  
If you like listening to podcast and want to listen to a cool story about how package indexes came to existence, I highly recommend the [CORECURSIVE episode 079  
"CPAN This Day In History"](https://corecursive.com/tdih-cpan/).  
  
The `uv publish` command takes an argument `[FILES]` to specify the paths of the distribution files you want to upload. It accepts glob patterns (`dist/*` by default) and automatically selects only wheel and sdist distributions, ignoring other file types.  
The options here are easy to understand as well, you have the `--publish-url` to specify the URL of the ***upload endpoint*** where the distributions will be sent. It defaults to <https://upload.pypi.org/legacy/>. This is not the index URL! Then you have the classic (`-u`, `--username`, `-p`, `--password`) or (`-t`, `--token`) for authenticating the upload. You can also use the keyring CLI for handling authentication credentials with `--keyring-provider`.  
You can also configure trusted publishing, which is useful when using GitHub Actions, using `--trusted-publishing`.  
Finally, another useful option which is `--check-url`. It checks the specified index URL to determine if a file already exists, thereby avoiding duplicate uploads. Before uploading, it verifies if the file exists on the index based on supported hashes (SHA-256, SHA-384, SHA-512). If a file exists, it skips uploading that file. If an upload error occurs, it rechecks the index to confirm whether the file was uploaded by another parallel process. It can also result a failed upload where some files were already successfully uploaded. The effectiveness of this option depends on the use case.



# EDITS



As pointed out by @not\_a\_novel\_account in this [comment](https://www.reddit.com/r/Python/comments/1gw1fe6/comment/ly6wo39/?context=3), I made an oversight in the `purelib` vs `platlib`. Firstly it's only the root which is either `purelib` or `platlib` (which should have been clear from the `Root-Is-Purelib` naming). Also, you might have `Root-Is-Purelib` being false while the root contains both `purelib` and `platlib`. He also gave a use case for why this distinction is helpful.  
  
Old text :  
"""  
`purelib` stands for pure Python library while `platlib` stands for platform-specific library.  
  
A package is either `purelib` or `platlib`, can't be both. `purelib` are packages that only contain pure Python code without any compiled extensions, while `platlib` are packages that include compiled extensions or platform-specific code.  
  
Why the distinction? `purelib` packages are installed in `site-packages`, while that might not be the case for `platlib`. It depends on the platform. They're also installed in `site-packages` but not the same path inside the `venv`. The [example from the specification](https://packaging.python.org/en/latest/specifications/binary-distribution-format/#what-s-the-deal-with-purelib-vs-platlib): "Fedora installs pure Python packages to ‘/usr/lib/pythonX.Y/site-packages’ and platform dependent packages to ‘/usr/lib64/pythonX.Y/site-packages’". Notice the `lib` vs `lib64`.  
  
That's why this distinction is made.  
"""
