+++
title    = "A Comprehensive Guide to Python Project Management and Packaging: Concepts Illustrated with uv - Part I"
date     = "2024-11-06T00:04:38+00:00"
draft    = false
categories = ["Python"]
tags       = ["popular", "Packaging", "uv"]
+++


The goal of this guide is to provide a comprehensive guide to Python project management and packaging.   
  
We'll explore concepts in the **standard**, like the different tables in `pyproject.toml` by revisiting the PEPs that led to what we have today. We'll explain what was used before, why it needed to change, and how the changes provided by the PEPs solved the issues. This walkthrough of the historical context is important to understand current practices.  
  
Since Python project management and packaging usually relies on tools (to install your dependencies, build your project etc.), I thought it was a great idea to use a tool to illustrate the concepts we talk about. In general tools go beyond the standard, for example the chosen tool here `uv` allows for what is called "**editable dependencies**", which is not something in the standard but a feature that many tools like `uv` offer. I find it important to delve in those extra features that are not supported by the standard but we find in many tools. On the other hand, when a standard provides a new feature, usually it's support is not immediate. For example with [PEP 735](https://peps.python.org/pep-0735), at the of writing, I was using `uv` version `0.4.29 (85f9a0d0e 2024-10-30)`, and it totally support the non-package use case yet, though it supported declaring dependency groups and it worked well for package projects.  
  
It's important to keep in mind this distinction between what is in the standard, what is beyond the standard and provided by a tool, and what new standards the tool lacks support of.  
  
[uv](https://github.com/astral-sh/uv) is "an extremely fast Python package and project manager, written in Rust". The commands we're going to cover are init, add, lock, sync, run, tool, build and publish. These commands will allow me to illustrate all the concepts we'll discover.  
  
There are many other similar tools such as [Poetry](https://github.com/python-poetry/poetry), [PDM](https://github.com/pdm-project/pdm), [pipenv](https://github.com/pypa/pipenv) etc. Don't hesitate to explore and try them and pick which suit your needs. [pip](https://github.com/pypa/pip) might be sufficient by itself for your needs. Again, the goal of this article is not to be a guide of a particular tool, but to get you from knowing nothing, to knowing the standard.  
  
What I won't get into in this article are scripts, workspaces (just a little mention in how to add them in the appropriate section), building a package and publishing. These will be in a second part of this guide.  
  
For an introduction to this topic in Turkish, you can read the following post: <https://www.sglbl.com/2025/01/uv-ve-pyprojecttoml-ile-python-projesi.html>



# uv init



The uv init command is self-explanatory, as are almost all of uv commands, it's used to initialize a project, let's check what it does when executed.  
  
Since we're starting from a blank project, this will create the following files: `.gitignore`, `.python-version`, `hello.py`, `pyproject.toml` and `README.md`. The most important files for us here are the `pyproject.toml` and to a lesser extent the `.python-version`.



The `.python-version` "contains the project's default Python version" and "tells uv which Python version to use when creating the project's virtual environment", from [the official documentation](https://docs.astral.sh/uv/guides/projects/#python-version).



Then there is the `pyproject.toml`.



## pyproject.toml






### High-level overview



Related PEPs: [518](https://peps.python.org/pep-0518/#tool-table), [621](https://peps.python.org/pep-0621/)  
Specifications: [pyproject.toml](https://packaging.python.org/en/latest/specifications/pyproject-toml/), [core metadata](https://packaging.python.org/en/latest/specifications/core-metadata/)  
PyPA guide: [Writing your pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)  
  
The `pyproject.toml` file is a standardized configuration file used to specify project metadata, dependencies, and build requirements in a declarative way. You can also use it to include tools other than build tools, and how they are configured.  
  
**Note**: having a `pyproject.toml` is not required for your project and there are other ways to specify build systems and metadata, but `pyproject.toml` became the standardized configuration file for Python project when it comes to project metadata, especially after [PEP 518](https://peps.python.org/pep-0518/) and [PEP 517](https://peps.python.org/pep-0517/).



### Details



In the following, I'll probably use interchangeably the words build system and build tool. But a build system is the whole program or set of programs that will turn your project into a distributable artifact (the `.whl` e.g.). The build tool is the CLI or the user interface part of it. And the build backend is the logic. It's like with Docker, you have the CLI and the Daemon doing the actual heavy lifting.



#### What did we have before pyproject.toml?



Before `pyproject.toml`, Python developers used an executable Python file, `setup.py`, to build, install and distribute their packages. And the main two ways to build a Python project were `distutils`, part of standard library, and `setuptools`, which isn't part of the standard library.  
  
`setuptools` appeared as a library that offered more features than what was provided with the standard library through `distutils`, but it also came with its lot of issues. You see, since `distutils` was part of the standard library, developers didn't require any dependencies (except for Python itself) to build their projects. A `setup.py` could look like (from <https://docs.python.org/3.11/distutils/setupscript.html>):



```python
#!/usr/bin/env python

from distutils.core import setup

setup(name='Distutils',
      version='1.0',
      description='Python Distribution Utilities',
      author='Greg Ward',
      author_email='gward@python.net',
      url='https://www.python.org/sigs/distutils-sig/',
      packages=['distutils', 'distutils.command'],
     )
```



As you see, there is nothing more needed than Python itself since `distutils` is part of the standard library. But `distutils` was pretty simple and though it was sufficient for projects with minimal needs, it didn't satisfy complex projects which required more functionality that `setuptools` provided. So it grew in popularty and developers preferred it, and you see the first issue we encounter? Now we have a **build dependency** which is `setuptools` to build or install the project. A `setup.py` using `setuptools` would look like (from <https://setuptools.pypa.io/en/stable/userguide/quickstart.html>):



```python
from setuptools import setup

setup(
    name='mypackage',
    version='0.0.1',
    install_requires=[
        'requests',
        'importlib-metadata; python_version<"3.10"',
    ],
)
```



This is what is referred to as "catch-22" in [PEP 518](https://peps.python.org/pep-0518/#tool-table) that introduced `pyproject.toml`. The "catch-22" problem refers to the circular dependency issue where to find out which dependencies are required to build a project you needed to run `setup.py` but to run `setup.py` you required `setuptools`, but to know the project required `setuptools` you had to run `setup.py` which would fail etc., **circular dependency issue**.



To specify the dependencies required to build a project, `setuptools` introduced the `setup_requires` argument to the `setup()` function. But it didn't solve the main issue of the circular dependency anyways, and now, we have many more issues.   
- For tools like `pip` to figure out your project dependencies they had to know what was in the `setup_requires` which required `setuptools`.   
- You couldn't use dependencies in your `setup.py`, in order to configure something before hand or whatever, before the `setup()` function itself because the dependencies listed in `setup_requires` were not installed until the execution of the `setup()` function.  
- And you could neither specify `setuptools` in the `setup_requires` (which makes sense since it's the library that processes that file anyways) nor specify other build tools (which also makes sense because they can't process the `setup_requires`). This is a big issue because you could not enforce a specify the `setuptools` version used which could lead to build failures for your users and if you updated to a new version of `setuptools` to benefit from some new features, there was no standardized and robust way to communicate that to your users.   
- To top it all off, dependency and package management tools like `pip` would also execute the `setup.py` file and so if you used the `setup_requires` argument, you'd usually run into this situation where both `pip` and `setuptools` executed the `setup.py` file and if you had different configurations for both tools (like from which index to pull packages) then you had this inconsistent behavior between both tools, potentially different dependency resolution, and a lot of different issues related to that.



*It was a mess*. And developers just ditched using the `setup_requires` and would copy paste whatever was in the `setup.py` of their external dependencies to avoid them, and would often write manual instructions to install the dependencies and build the project. This led `pip` to assume `setuptools` was required if you had a `setup.py`, though it still didn't solve every issue above and it wasn't beneficial for the community since it was a kind of barrier to entry to new build systems.



And in 2016, [PEP 518](https://peps.python.org/pep-0518/#tool-table) introduce `pyproject.toml` as a new way to declare project and build system dependencies that solves the above issues.



#### Why the way of pyproject.toml solved the issues?



If you think about it and you think of the comparison between `setup.py` and `pyproject.toml`. We have on one hand a dynamic, executable file that contains imperative code that is ran when using commands, and on the other hand we have a static, declarative file with no executable code but only text in a format that is human-readable and easily parseable and interpreted by tools and build systems. We have on one hand code, and on the other hand data.



What made `setup.py` not scale beyond the limited uses with `distutils` is that it required code to be executed. And as long as you have needs that aren't spported by the standard library, then you're going to run into issues. On the other hand, `pyproject.toml` is simply a static file that any tool, any build system can read, parse and interpret, regardless of the code the program was written in. You don't require anything to discover what is inside a `pyproject.toml` and that was what made it so great. (By the way, as it is mentioned in [PEP 518](https://peps.python.org/pep-0518/#tool-table), other programming languages used similar ways with their dependency and package management tools like Rust with Cargo)



Now tools like `pip` can read this and install the necessary build tools before building the project, it works perfectly with other tooling, you can use different build systems and the issue of circular dependency is solved.



The [PEP 518](https://peps.python.org/pep-0518/#tool-table) also goes into some decisions for the `pyproject.toml` like why the TOML format, the naming, the rejection of a semantic version key etc.



#### What goes into a pyproject.toml?



A `pyproject.toml` can be comprised of **three tables**, `[project]`, `[build-system]`, and `[tool]`. **They're not mandatory** to have but generally a `pyproject.toml` contains at least the `[project]`table. That's what the simple `uv init` command generated for us for example. We'll see later on that the `init` command provides options that automatically add the `[build-system]`table as well. As for the `[tool]`table then other commands act on it like `uv add --dev` or when specifying configurations for your tools, like Ruff.



[PEP 518](https://peps.python.org/pep-0518/) specifies the table `[build-system]`with one mandatory key, `requires`. Which must be an array of strings **specifying [PEP 508](https://peps.python.org/pep-0508/)-compliant dependencies required to build the project**. Examples: `requires = ["setuptools", "wheel"]` or `requires = ["hatchling"]`.  
"Initially only one key of the table will be valid and is mandatory for the table: `requires`" (quoted from [PEP 518](https://peps.python.org/pep-0518/)) but [PEP 517](https://peps.python.org/pep-0517/) introduced an additional key to this table, it's the `build-backend` key, which **specifies the build backend to use for building the project**. Example: `build-backend = "hatchling.build"`.



[PEP 518](https://peps.python.org/pep-0518/) also specifies the goal behind the `[tool]`table. It's reserved for configuring third-party tools. Each tool should have its own sub-table `tool.$NAME`. Example: `[tool.uv]` or `[tool.ruff]`. You can also more depth in these sub-tables like `[tool.ruff.lint]`and `[tool.ruff.format]`.



While [PEP 518](https://peps.python.org/pep-0518/#tool-table) introduced the `[build-system]` and `[tool]` tables in `pyproject.toml` to specify a Python project's minimum build system requirements and tool-specific configurations, [PEP 621](https://peps.python.org/pep-0621/) specifies how to declare the core metadata of a project through the `[project]`table.   
  
There are only two mandatory keys, `name` which is the name of the project, and `version` which is the version of the project as outlined in [PEP 440](https://peps.python.org/pep-0440/). `name` **must be statically provided** while `version` **might be provided dynamically**, either way, it's a **required key**.  
The optional keys specified by this PEP are: `description`, `readme`, `requires-python` (it's **recommended** to have it though), `license`, `authors`/`maintainers`, `keywords`, `classifiers`, `urls`, `entry-points`, including `scripts` and `gui-scripts` (we'll talk about entry points later on), `dependencies` and `optional-dependencies` and `dynamic`. I suggest you read their description and what the PEP says about them to know more about how you should specifiy them.  
  
**What is the `dynamic` key?** It's an array of strings that is used to list the keys that were **intentionally left unspecified** and that are **expected to be provided dynamically** by the build backend. That's an important behavior to understand.   
- The keys that are left unspecified and that are not mentioned in `dynamic` can't be filled by a build backend.  
- `name` and `version` are not optional (they can't be left unspecified). `name` can't be specified in `dynamic`, it must be statically provided. `version` can be specified in `dynamic`, so if it's not statically provided it must be dynamically provided, it's a required key.  
  
The data specified statically in `[project]` is "**canonical**". It's a source of truth and tools / build backend must use it as it is, they can't "remove, add or change" it. Only the dynamic data is allowed to be altered for a tool to bring a "new“ value to it. Tools can and must raise an error if the statically specified metadata is inappropriately specified (for example providing a version that's not [PEP 440](https://peps.python.org/pep-0440/) compliant).  
And in general the `pyproject.toml` is kind of an authoritative source of truth, I think it's one of the fundamental ideas behind it, that led to the rejection of allowing build backends to alter it when generating an sdist (we'll get into that later on).  
  
The `dynamic` key offers an "espace hatch" as mentioned in the PEP. It's allows for **partial opt-out** of the PEP by only specifying statically some parts of the PEP and using a build backend to provide dynamically the rest. You can also **fully opt-out** of the PEP by not having the `[project]` and letting a build backend to provide everything. **But**, many build backend would just fail and announce that they require the project table.  
  
There is this important distinction between static and dynamic because the PEP encourages to have as much static data as possible. It keeps the metadata deterministic with respect to the tools, while it's also simple and clear. Remember that the main advantages of a `pyproject.toml` is to have a standardized way of specifying project metadata in a human-readable format.  
  
On top of that, tools can't add fields of their own to the `[project]`table. Whatever else they want to bring is to be specified in the `[tool]`table. That's why for example if you do uv add `--editable ../my_lexer_package` you'll get something different from doing `uv add openpyxl --optional excel`. The following are respectively the additions from `uv` to the `pyproject.toml`:



```
[project.optional-dependencies]
excel = [
    "openpyxl>=3.1.5",
]

[tool.uv.sources]
my-lexer-package = { path = "../my_lexer_package", editable = true }
```



It's because, there is no support for path sources in the standard but it's a feature that many tools implement. `uv`, or any other tool for that matter, doesn't have the right to add a `sources` key to the `[project]`table.



**Note**: the `[project]`table, as mentioned in the PEP, hold the **project metadata** and not the **build metadata**. Project metadata are all the relevant information about a project and tools that interact with it. They're specified by the [core metadata](https://packaging.python.org/en/latest/specifications/core-metadata/). [PEP 621](https://peps.python.org/pep-0621/) only **specifies a subset of the core metadata** and **how it should be used in the** `pyproject.toml`.  
Build metadata is all the information that a build system will using during its build process. It can be configuration files, compilation options, instructions for which files to include and which not to include etc. As said by the authors in the PEP, the goal is to standardize the project metadata in order to provide a consistent way to specify metadata that describes the project. Not specifying build metadata allows build systems to be flexible to handle build configurations and files to their taste. It also allows for the `pyproject.toml` to remain fairly simple.  
By not touching these areas, this PEP allows allows for experimentation and innovation in the areas that lack some kind of pre-existing consensus while also allowing for the `pyproject.toml` to be clear and focused. The goal is also to have an "ergonomic TOML" file. So not only the PEP doesn't try to standardize build metdata but also doesn't try to "mirror" how build backend represent the metadata which might not be uniform across build backends and might be verbose and not suitable for clarity and human reading.



**Note**: There's another table in `pyproject.toml` called `dependency-groups` and is defined by [PEP 735](https://peps.python.org/pep-0735/). We're not mentioning it here to get into its details later on in the `add` and `remove` commands.



#### What are entry points?



Specification: [Entry points specification](https://packaging.python.org/en/latest/specifications/entry-points/#entry-points)



According the specification, "*Entry points* are a mechanism for an installed distribution to advertise components it provides to be discovered and used by other code". The definition is a bit vague, but it just means that **entry points are metadata definitions** in a Python package that **specify** interfaces like having a **CLI** tool or **GUI** interface, or features (more like **plugins**) provided by the package for other packages. Plugins are useful because they allow other packages to discover and use them (we'll soon see how) without having to hard-code the imports.   
  
So entry points are a way to advertise this information and show how to use it, if your package provides a **CLI** tool, you'd like to tell that information to the tools that can use it, like `pip`, and tell them how to use it, so you have to provide them with a **name** for the CLI, and an **object reference** which is the like a relative path to the function which will be called with no arguments when your command is run. For example, [pygments](https://github.com/pygments/pygments/tree/master) has the following [entry point](https://github.com/pygments/pygments/blob/ce53c4e4b7548fb97d8c532bec82f8d275d9a7f1/pyproject.toml#L55) `pygmentize = "pygments.cmdline:main"` which means that it offers a CLI named `pygmentize`, and the function that should run when `pygmentize` is called is the `main` function found in the module `pygments.cmdline`. [Here is its code](https://github.com/pygments/pygments/blob/ce53c4e4b7548fb97d8c532bec82f8d275d9a7f1/pygments/cmdline.py#L528).  
When we write such an entry point, doing `pip install Pygments`, `pip` will wrap that `main` function in a CLI named `pygmentize`.   
  
Each entry point is composed of a **group**, a **name** and an **object reference**. The group determines the category or the type of the entry point, it indicates whether the entry point is a CLI, a GUI or a plugin. The name is an identifier within the group, it can be the name of the CLI or it can identifies the plugin. The object reference a Python import path to the object (module, function or class) that implements the interface or plugin. It has to be either in the format `importable.module` or `importable.module:object.attr`.  
  
There are three types of entrypoints:



- **Command-Line Interface (CLI)**:
  - Group (in `pyproject.toml` according to PEP 621): `scripts`. So you declare your CLIs in `[project.scripts]`.
  - The object reference must to point to a function that can be made available for execution as a command in the terminal. When tools install your package, like `pip`, they'll wrap these functions and make them available as commands.
  - The CLIs will not necessarily be available system wide. It all depends on the environment and how Python is installed. If you `pip install` a package that has a CLI entry point in a virtual environment, you won't be able to access it unless using that virtual environments. Tools don't modify `PATH`. In this case the executable will be placed in the `bin` within the virtual environment.
- **Graphical User Interface (GUI)**:
  - Group (in `pyproject.toml` according to PEP 621): `gui-scripts`. So you defien the GUI apps in `[project.gui-scripts]`.
  - Everything is similar to a CLI except that you launch a GUI app instead of using a command.
- **Plugin**:
  - Group (in `pyproject.toml` according to PEP 621): plugins must be declared under the `entry-points` table. They're constrained to a depth of 1 as sub-tables, so you can't do something like `[project.entry-points.text_plugins.tokenizers]` (imagine you have different groups of text plugins). In this case `text_plugins.tokenizers` is a nested table. You have to do `[project.entry-points."text_plugins.tokenizers"]`. With this latter form, "text\_plugins.tokenizers" are treated as one key instead of having a nested table `tokenizers` within the table `text_plugins`.
  - The decision to restrict the `entry-points` to table of one level depth was made for two main reasons:
    - Future proof the structure. Allowing arbitrary depth tables could interfere with adding additional fields to `project.entry-points` for specifying metadata for example or something else.
    - Simplify the task for build tools. Build tools won't have to traverse the entire nested table to figure out the entry-point. This simplifies the parsing.
  - Think of plugins as offering add-ons to other packages. Example: [pygments](https://github.com/pygments/pygments), which is "a **generic syntax highlighter** written in Python", has a [mapping](https://github.com/pygments/pygments/blob/ce53c4e4b7548fb97d8c532bec82f8d275d9a7f1/pygments/lexers/_mapping.py#L4) of 584 (as of version 2.18.0) lexers. But what if you want to add your own lexer? You can't expect pygments to add it manually or hard code the import for it. What you can do is write your lexer as a plugin, specify it under the `pygments.lexers` in the `entry-points` and pygments will discover it. It calls this function [find\_plugin\_lexers](https://github.com/pygments/pygments/blob/ce53c4e4b7548fb97d8c532bec82f8d275d9a7f1/pygments/plugin.py#L55C5-L55C23) that will discover all available pygments lexers plugins (if you check the code it relies on `LEXER_ENTRY_POINT = 'pygments.lexers'`, that's why your plugin should be in the `pygments.lexers` group). The main function behind discovering entry points is `importlib.metadata.entry_points` but this page "[Creating and discovering plugins](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/)" will provide you with more information.
  - **Why are plugins cool?** They allow to extend the functionality of a library or a package without having to modify it.



**Note**: With `uv`, both CLI and GUI entry points require a build system to be defined.



In case you're still struggling with the plugin entry points, try making one yourself for a library or a package. Below we'll make a toy plugin for `pygments`.



So I'll create two projects, the first one where I create the plugin, `my_lexer_package`, and the second one where I test it, `test_my_lexer`.  
  
The structure of `my_lexer_package` is going to be:  
.  
├── my\_lexer\_package  
│   ├── **init**.py  
│   └── my\_lexer.py  
├── pyproject.toml  
└── setup.py  
  
The structure of `test_my_lexer` is going to be like:  
.  
├── list\_lexers.py  
└── test\_lexer.py



Let's start with the `pyproject.toml` for `my_lexer_package`. My only dependency is `pygments` since it'll allow me to create a lexer. I'll use `setuptools` as build backend. My entry point is a plugin for `pygments` so I'll put it in the `pygments.lexers` group, the code for it will be the `MyLanguageLexer` class in `my_lexer_package.my_lexer`.



```
[build-system]
requires = ["setuptools>=75.0"]
build-backend = "setuptools.build_meta"

[project]
name = "my_lexer_package"
version = "0.1.0"
description = "A custom toy lexer for Pygments"
authors = [{ name = "Name", email = "email@email.com" }]
dependencies = ["pygments"]

[project.entry-points."pygments.lexers"]
mylanguage = "my_lexer_package.my_lexer:MyLanguageLexer"
```



The code for `my_lexer.py`:



```python
from pygments.lexer import RegexLexer
from pygments.token import Keyword, Name, Text


class MyLanguageLexer(RegexLexer):
    name = "MyLanguage"
    aliases = ["mylanguage"]
    filenames = ["*.my"]

    tokens = {
        "root": [
            (r"\b(if|else|for|while)\b", Keyword),
            (r"[a-zA-Z_]\w*", Name),
            (r"\s+", Text),
            (r".", Text),
        ],
    }
```



And the simple `setup.py`:



```python
from setuptools import setup

setup()
```



Now, I go to my `test_my_lexer` folder and I run these commands:



```
uv init
uv add pygments --editable ../my_lexer_package
```



You'll get a warning for lower bounds on versions but it's okay for this toy example. You should get a `pyproject.toml` like the following:



```
[project]
name = "test-my-lexer"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "my-lexer-package",
    "pygments>=2.18.0",
]

[tool.uv.sources]
my-lexer-package = { path = "../my_lexer_package", editable = true }
```



Now let's put in two Python scripts to test our lexer, `test_lexer.py`



```python
from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import get_lexer_by_name

code = """
if condition:
    do_something()
else:
    do_something_else()
"""

lexer = get_lexer_by_name("mylanguage")
formatter = TerminalFormatter()
result = highlight(code, lexer, formatter)
print(result)
```



Executing `uv run test_lexer.py` should show you the code string with the if/else highlighted. We can also check that our lexer gets correctly discovered by doing `uv run list_lexers.py | grep MyLanguage` where `list_lexers.py` is the following:



```python
from pygments.lexers import get_all_lexers

for lexer in get_all_lexers():
    print(lexer[0])
```



## Back to uv init



There are different options for `uv init`. They're all pretty self-explanatory as well, like `--no-readme` which, when used, doesn't create a `README.md` file in the initialized project. But I want to delve into four options that I believe are important for understanding Python packaging. They're `--package`, `--no-package`, `--app` and `--lib`. Let's see what the manual says about them:



```
      --package
          Set up the project to be built as a Python package.
          
          Defines a `[build-system]` for the project.
          
          This is the default behavior when using `--lib`.
          
          When using `--app`, this will include a `[project.scripts]` entrypoint and use a `src/` project structure.

      --no-package
          Do not set up the project to be built as a Python package.
          
          Does not include a `[build-system]` for the project.
          
          This is the default behavior when using `--app`.

      --app
          Create a project for an application.
          
          This is the default behavior if `--lib` is not requested.
          
          This project kind is for web servers, scripts, and command-line interfaces.
          
          By default, an application is not intended to be built and distributed as a Python package. The `--package` option can be used to
          create an application that is distributable, e.g., if you want to distribute a command-line interface via PyPI.

      --lib
          Create a project for a library.
          
          A library is a project that is intended to be built and distributed as a Python package.
```



To understand well what's the difference between these options, we have to understand what's an **application**, a **package** and a **library**. Let's delve into this nomenclature.



## Python packaging nomenclature






### High-level overview



`--app`: sets up a Python **project** with no intention behind (whether it's going to be a package or not). A project is a collection of code and resources, under development. It can contain test code, docs, configs etc. The purpose of a project is to be distributed later on but `uv` assumes differently.  
  
The default behavior of `--app` is `--no-package` because an application at this point is not assumed to be something else. With `--package` you assume it's going to be a **distribution package**. A distribution package is something you can install, it's something you push to PyPI and users can install with `pip` for example. It can be a `.whl` or a `.tar.gz`. But here `uv` considers that an application is not going to be distributed. And when you do `--package`, `uv` assumes that your application is going to be a specific type of packages, which is the type of web servers, scripts and CLIs. So when you do `--app --package` you signify to `uv` that you're developing something like a CLI that you're going to distribute later on, so both the `pyproject.toml` and the structure of your directory change.  
  
`--lib` is something that you import, it contains code. Often you can package it into a distribution package and often you acquire it by installing a distribution package. It's not a CLI so `uv` won't bother with entry points, but you're still going to distribute it, thus the default behavior of `--lib` is `--package`.



Concretely, with `uv init` you mean `uv init --app` which also means `uv init --app --no-package`. You get a simple `pyproject.toml`:



```
[project]
name = "just-a-project"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = []
```



And executing `uv init --package` is the same as `uv init --app --package`, it sets up your project to be a distribution package. You get the following `pyproject.toml`:



```
[project]
name = "to-be-package"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
authors = [
    { name = "Name", email = "email@email.com" } 
]
requires-python = ">=3.13"
dependencies = []

[project.scripts]
to-be-package = "to-be-package:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```



`uv` infers the authors fields from the VCS. It also assumes that your package is going to contain a CLI. And as you have guessed, it adds a `src/to-be-package` directory.



Finally, `uv init --lib` is the same as `uv init --lib --package` and `--lib` doesn't work with `--no-package`, and obviously you can't use `--lib` and `--app` at the same time. You get a similar structure as with `--package` alone, but `uv` doesn't assume a CLI entrypoint and adds a `py.typed` file in your `src/project`:



```
[project]
name = "to-be-lib"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
authors = [
    { name = "Name", email = "email@email.com" } 
]
requires-python = ">=3.13"
dependencies = []

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```



### Details



If you want a complete set of Python packaging glossary, visit <https://packaging.python.org/en/latest/glossary/>  
Here I'll only focus for what's needed at the moment.  
  
**Module**: "The basic unit of code reusability in Python". It's either a `.py` file, and in this case it's called a **pure module** (it can be anything even as simple as defining a function that prints hello world) or something written in a low-level language (generally C because CPython is the most common implementation of Python) and in this case it's called an **extension module**.  
  
**Package**: people generally use the term package to talk both about an "import package" and a "distribution package".   
An **import package** is something you can import. It contains modules or recursively, other import packages. It's a way to structure the Python's module namespace with dotted module names. For example the `torch` import package, contains the subpackage `nn`, which contains modules (single `.py` files) and a subpackage `modules` which only contains modules (single `.py` files) like `conv.py`.   
A **distribution package** is something that you can install. It's a versioned archive file that contains Python import packages, modules, and other resource files, along with metadata like version numbers and dependencies. It's what you typically install with `pip install` or what `uv add` installs.  
There's obviously a relationship that leads to people using the same term "package" for both. Usually you do `pip install numpy` and then in your Python modules you do `import numpy as np`.  
Import packages and distribution packages don't always have the same name, [Distribution package vs. import package](https://packaging.python.org/en/latest/discussions/distribution-package-vs-import-package/) talks about how the `Pillow` distribution package (`pip install pillow`) provides the PIL import package (`from PIL import Image`).The `--package` option for the `init` command means a distribution package. When using it with `--app` it means you're going to distribute a CLI-like package, and when doing `--lib` (then the `--package` is the default behavior) it means you're going to distribute something that will be used as an import package.  
  
**Project**: this term is very vague, but it just means **all the code and resources that are intended to be distributed as a (distribution) package**. But it's still in an "not built" yet state.  
  
Example:   
- When you `pip install pandas` then you can say you're installing the (distribution) package `pandas`.   
- When you friend tells you to import `DataFrame` from `pandas`, he's talking about the (import) package `pandas`.  
- If you're contributing to `pandas`, then you're contributing to the `pandas` project, and hopefully your contributions will be in the next release of the `pandas` (distribution) package!



# uv add & uv remove



Since `uv add` and `remove` deal with dependencies, this section will talk about the relatively newly accepted [PEP 735](https://peps.python.org/pep-0735/). To benefit from that with `uv`, make sure you have at least the [0.4.27](https://github.com/astral-sh/uv/releases/tag/0.4.27) release, or a later release.  
  
 I believe there are two main parts for how these commands interact with your project. The first part is how you use them to edit the `pyproject.toml`, and that part is only concerned with dependencies. The second part is how these commands will affect your lockfile and virtual environment.



## Dependencies






### High-level overview



There 4 types of dependencies that you should know about when it comes to Python projects. The normal project **dependencies**, the **optional dependencies**, the **development dependencies** and **dependency groups**, and finally the **editable dependencies**.  
  
Since [PEP 735](https://peps.python.org/pep-0735/), many tools, like `uv`, include development dependencies as a dependency group.



- Project **dependencies**:
  - They are just what you're used to, the essential packages that your project requires to function properly.
  - Inclusion in `pyproject.toml`: as a [PEP 508](https://peps.python.org/pep-0508/) compliant string in the `dependencies` key (which is an array of these strings) in the `project` table.
  - How to add them using `uv`: `uv add <package> ...`
  - Example: `uv add numpy pandas`
  - Standard: [PEP 621](https://peps.python.org/pep-0621/)
  - Non-included in the standard: A particular case of project dependencies are **editable dependencies**. These are packages that are under development, and that you want to include in an **editable** mode, which is a mode that will reflect any changes made to these dependencies immediately in your project. You can add them by using uv: `uv add --editable <package> ...` where `package` is the path on your filesystem to the dependency. `uv` will add the package to `dependencies` and will also add this `<package-name> = { path = <package-path>, editable = True}` in the `tool.uv.sources` table `tool.uv.sources`. And as pointed out by this [comment](https://www.reddit.com/r/Python/comments/1gkmrsg/comment/lvofx3m/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) (from u/ArchFFY00), there is no disparity in this feature between the different tools and this (the inclusion of editable dependencies) relates to the lockfiles which are not covered by the standard. As mentioned, an editable dependency *is* a dependency, so it's included in the `dependencies` of a project, it's just *how* the tool includes that dependency as an editable one that will differ from one tool to another.
- **Optional dependencies**:
  - They are, as the name suggests, optional dependencies, not essential for the package to run, but are essential to enable certain features. They help reduce the default dependency tree. For example if you look at Pandas' `pyproject.toml` you [see](https://github.com/pandas-dev/pandas/blob/9a015d19514597ad5d1c31f566a0a2bdb17d4cc5/pyproject.toml#L68C1-L68C124) `excel = ['odfpy>=1.4.1', 'openpyxl>=3.1.0', 'python-calamine>=0.1.7', 'pyxlsb>=1.0.10', 'xlrd>=2.0.1', 'xlsxwriter>=3.0.5']` in its `optional-dependencies`. So if you do `pip install "pandas[excel]"`, you install the base `Pandas` package as well as the packages in the `excel` **extra**.
  - **Extras** are what the "groups" in `optional-dependencies` called.
  - Inclusion in `pyproject.toml`: as a sub-table of the `project` table, named `optional-dependencies` where each of its key specifies an extra and whose value is an array of [PEP 508](https://peps.python.org/pep-0508/) compliant strings.
  - How to add them using `uv`: `uv add --optional <extra> <package> ...`
  - Example: `uv add --optional excel openpyxl`
  - Standard: [PEP 621](https://peps.python.org/pep-0621/)
- **Dependency groups**:
  - Dependency groups are dependencies organized into logical groups, such as development, testing, documentation etc. **These groups are not included in the package's published metadata** and are primarily for development and project organization.
  - Inclusion in `pyproject.toml`: in the `dependency-groups` table, each group is a key for which the value is an array of [PEP 508](https://peps.python.org/pep-0508/) compliant strings.
  - How to add them using `uv`: `uv add --group <group> <package> ...`
  - Example: `uv add --group testing pytest coverage`
  - Standard: [PEP 735](https://peps.python.org/pep-0735/)
- **Development dependencies**:
  - Dependencies during the development and are not essential to the project, like testing packages, documentation etc.
  - Since [PEP 735](https://peps.python.org/pep-0735/), they're better thought of as a group in dependency groups.
  - You should keep in mind that many of the groups in your dependency groups can be thought of as development dependencies, but tools allow to separate and have a specific term for development packages such as `dev`.
  - Standard: there is no standard for development dependencies but since [PEP 735](https://peps.python.org/pep-0735/) they can be put inside the `dependency-groups` table as the `dev` group.
  - How to add them using `uv`: `uv add --dev <package> ...`
  - **Note**: `--dev` is an alias `--group dev`.
  - Before [PEP 735](https://peps.python.org/pep-0735/), `uv` used to put the development dependencies inside the key `dev-dependencies` of its tool table (`tool.uv`).



**Note**: there are also build dependencies, but that's not really the scope of this part.



**Note**: `uv remove` works the same way as `uv add` except it doesn't have an `--editable` option. It's not possible to remove an extra or a group completely. But you can remove packages from them. So there is a perfect parallel between `uv add` and `uv remove`. If `uv add --dev <package>` adds `package` to the `dev` dependency group, then `uv remove --dev <package>` removes `package` from the `dev` dependency group.



### Details



#### Why dependency groups?



The [PEP 735](https://peps.python.org/pep-0735/) which brings to the standard the dependency groups identifies two issues for which there is no standardized way to deal with and explains how it solves the shortcomings of the two main ways people rely on to solve those issues.   
  
There are two types of Python projects are affected by those issues, the project that are intended to be built as that the PEP calls packages, and the non-packages which are the projects not intended to be built. This distinction is important because each issue targets a different kind.  
  
The first issue is **how to declare development dependencies for packages**.  
The second issue is **how to declare dependencies** (whether they're for development or not) **for non-packages**.  
  
The two mains ways of dealing with these are `extras` (the optional dependencies in `optional-dependencies`) and `requirements.txt` files.



##### What is already in the standard for these issues? (before PEP 735)



- For the first one, there isn't really something specific for development dependencies for packages.  
- For the second issue, you can use the `project` table to declare dependencies and optional dependencies. But many tools assume that if there is a `project` table then the project is intended to be built, and sometimes you just don't want to bother with that, you just want a way to declare different kinds of dependencies, imagine an equivalent of `requirements.txt`, what if we wanted a `pyproject.toml` that acts like a `requirements.txt`.



##### What are the ways people solved the first issue and why it's unsatisfactory?



- Many developers use the `optional-dependencies` from the `project` table, the `extras`. Why it's problematic:
  - Installing `extras` require installing the project's dependencies as well, because they're additional dependencies, so it only makes sense to have them if you have the project. They extend the project. And there are cases where you don't want that, you just want a simple development environment to run some checks or formatting etc., or maybe you're working on the documentation etc.
  - The second issue is that the optional dependencies are part of the package's public metadata when you publish it, which makes sense. If you're installing `pandas` you'd like to know which optional packages you can add. But if you're developing `pandas` it doesn't really make sense to push to the public the packages you're using for your development. It can be confusing to see them alongside the rest of the optional packages and it is unnecessary.
  - Finally, this way of doing might lead to issues due to non statically defining your development dependencies. Because the `extras` are part of the package metadata, so they might be left to the build system to resolve them. But you don't want that for your development dependencies.



**Note**: Some people also use `requirements.txt` files but it's rare when you're working on a package since generally you also have many things such as build dependencies etc., so people tend to use the `pyproject.toml` as much as possible.



##### What are the ways people solved the second issue and why it's unsatisfactory?



- Many developers use the `requirements.txt` files. Why it's problematic:
  - There's no standard whatsoever for these files. You can name them however you want, you can put them wherever you want. No issues. So that causes a lot of headache for tools to discover them. Some projects might have `requirements.txt`, `requirements-dev.txt`, `tests/requirements.txt` etc. Even beyond tools, you can see how it's not "scalable" to have different files with different names, scattered across a project, even for humans.
  - On top of that, `requirements.txt` files don't have a standardized content. They're much more tuned for `pip`. In a `requirements.txt` you can include `pip` options, so other tools can't necessarily process them. And not only that, even the data itself that these file contain might not be [PEP 508](https://peps.python.org/pep-0508/) compliant, they might contain hashes for package verification etc.
  - So there is a high cost when it comes to working with `requirements.txt` files because you have to create each file for each dependency, come up with a naming convention to help you work with the names/locations, and it requires more effort to maintain these files and keep them updated. Having a concise way to declare these dependencies would be very advantageous and that's what [PEP 735](https://peps.python.org/pep-0735/#example-dependency-groups-table) brings.



**Note**: It's possible to use the `optional-dependencies` from the `project` table, in this case, but I don't delve into it because I didn't see many people do that. It requires from you to declare a `project` table, which you don't necessarily want to do with a non-package project, and then some tools might think that your project is a package when the `project` table is present which will lead them to try and validate it (so you'll have to at least add the `name` and `version` keys).



##### How does PEP 735 overcome the shortcomings of the previous ways?



So we found that we **need a simple way to specify one group** (e.g., development dependencies) **or many groups** (e.g., case of a non-package project) **of dependencies** in such a way that we can **declare them without needing a project** (this is for non-package projects), **install them without having to install anything else**, like the dependencies of the project, (this is for both package and non-package projects), and having them as part of `pyproject.toml` **without publishing them as public metadata of the package** (this is for package projects).  
  
That's what [PEP 735](https://peps.python.org/pep-0735/) brings with its `dependency-groups` table.  
  
Let's see how dependency groups solve all the limitations of the previous solutions:



- Overcoming the limitations of `requirements.txt`:
  - Issue: Lack of naming / location standardization. 
    - Dependency groups have one clear location, it's a table within the `pyproject.toml`.
  - Issue: Non-standardized content / pip-specific options in the files.
    - The groups in the dependency groups can be either a list of [PEP 508](https://peps.python.org/pep-0508/) compliant strings for declaring dependencies, which all tools are able to process, and [PEP 735](https://peps.python.org/pep-0735/) introduces a new specification, called [Dependency Object Specifiers](https://peps.python.org/pep-0735/#dependency-object-specifiers).
    - **Dependency Object Specifiers** are just tables (or mappings if you prefer) that specify one or a set of dependencies. For the moment, the only type of Dependency Object Specifiers that is introduced in the PEP is the [Dependency Group Include](https://peps.python.org/pep-0735/#dependency-group-include). The goal of the Dependency Group Include is to offer a feature that is similar to having the `-r` flag inside a `requirements.txt` (which allows to include a different `requirements.txt` file in the one that contains the `-r requirements.txt`). So Dependency Group Includes being Dependency Object Specifiers, they're tables, and their syntax is `{include-group = "group"}`, which **allows one dependency group to include another**.
    - This is not to be confused with **Dependency Specifiers** from [PEP 508](https://peps.python.org/pep-0508/) which are strings that specify a package and optional version constraints, environment markers, and extras etc. (like `"requests>=2.25.1"`).
    - When a dependency group includes another one, then it is implicitly expanded when installed, e.g. the `extended` group in the following  
      `[dependency-groups]   
      base = ["packageA", "packageB"]   
      extended = [ {include-group = "base"}, "packageC" ]`  
      Is the same as `extended = ["packageA", "packageB", "packageC"]`.   
      Dependency Group Includes are expanded in their place.  
      A dependency group containing a cycle is an invalid dependency group and tools must report it as an error. E.g.:  
      `[dependency-groups]   
      group1 = ["package1", {include-group = "group2"}]   
      group2 = ["package2", {include-group = "group3"}]   
      group3 = ["package3", {include-group = "group1"}]`  
      You can also check that the [reference implementation](https://peps.python.org/pep-0735/#reference-implementation) reports an error in this case when it finds a cycle (see `_resolve_dependency_group`).
  - Issue: High `requirements.txt` files creation and maintenance costs.
    - It's extremely easy to keep track of, add, remove and update groups in dependency groups.
  - Issue: Portability problems with `requirements.txt` being a bit tailored to `pip`. 
    - Dependency groups are just declared data in `pyproject.toml` with a specification not being tailored for any tool (it's just lists of dependency specifiers and dependency object specifiers) so it's easy for tools to add support for them.
- Overcoming the limitations of `extras`:
  - Issue: being tied to package metadata. 
    - `dependency-groups` are not part of the package metadata and are not included in built distributions.
  - Issue: installing `extras` requires installing the package.
    - A dependency group can be installed independently from the package (even if the package is non-existing, like in the case of non-package projects like data science projects) and from other dependency groups.
  - Issue: `extras` being part of the package metadata they might lead to being dynamically defined, which can make them require a build system for the resolution.
    - Dependency groups are not part of the package metadata so they have to be statically defined which removes the need for having a build system to do the resolution.
  - Issue: not suitable for non-package projects.
    - Dependency groups can live independently from any other table in the `pyproject.toml`.
  - Also just semantically it's good to separate the `extras` which are optional dependencies that extend your package from the rest of dependencies that you might want (documentation, testing, formatting etc.).



**Bonus points** for dependency groups: they work seamlessly for both package and non-package projects. At least, in the standard. When it comes to tools, there might be variations. We'll discuss what is expected from tools in the following section. At the time of writing, with `uv` version `0.4.29 (85f9a0d0e 2024-10-30)`, the non-package use case was not supported yet. You had to include a `project` table (so at least have a `name` and `version` keys) or use the following [workaround](https://github.com/astral-sh/uv/issues/8778#issuecomment-2453524119):



```
[dependency-groups]
group_name = ["packageA"]

[tool.uv.workspace]
```



But the package project use case was completely supported.



##### Dependency groups and lockfile generation



I advise you to read the [use cases](https://peps.python.org/pep-0735/#appendix-c-use-cases) in the PEP, and the [rejected](https://peps.python.org/pep-0735/#rejected-ideas) ideas, it gives a solid understanding of the design decisions. If anything the rejected ideas are my favorite sections of the PEPs. Maybe the [deferred ideas](https://peps.python.org/pep-0735/#deferred-ideas) as well if you want to stay at the edge of what's happening.  
  
But I wanted to tackle the [lockfile generation](https://peps.python.org/pep-0735/#lockfile-generation) use case because lockfiles are important files in your project. And we'll see many commands that interact with them (namely the `add` and `remove` commands) and how they interact with them by default and through some options.  
  
**What are lockfiles?** lockfiles are files that record the exact resolved versions of packages and their dependencies that are installed in an environment. `uv` generates cross-platform (universal) lockfile named `uv.lock` while Poetry generates a platform agnostic lockfile called `poetry.lock` and a `requirements.txt` file with pinned versions and hashes such as generated by `pip-tools` can be considered a lockfile as well.  
Lockfiles as very important since they ensure consistency and reproducibility across environments.  
  
We can see that dependency groups can't store lockfiles (and it's not their purpose anyways) but they can be used as inputs to **lockfile generation**.  
The PEP discusses how tools might have an interface to do something like: `$TOOL lock --dependency-group=test which will prompt the $TOOL` to resolve the `test` group dependencies and sub-dependencies and generate a lockfile for it.   
The PEP also discusses how we can use that lockfile to install the group: `$TOOL install --dependency-group=test --use-locked` so that **only that group is installed**.  
At the time of writing `uv` doesn't support this kind of selective locking yet but it's really useful to save time and resources.  
  
One caveat though is when different groups aren't compatible then you might generate different lockfiles, but you won't be able to install them in your project environment.  
  
The PEP suggests two strategies of handling this issue:  
- **Mutual compatibility requirement** **per combination**: Require a valid lockfile generation for the combination of groups that is problematic. For example if `groupA` and `groupB` are conflicting, then the tool will require from you to solve the issues so that it can generate a lockfile first for the combination `groupA, groupB`, before you attempt to install them both. So, you can generate the lockfiles individually, you can install the groups individually, but if you want to install a combination, then you must generate a valid combination lockfile first in case the combination contains conflicting groups. The tool won't try to use the individual lockfiles.   
- **Global compatibility requirement**: Fix the issues across all groups first to ensure they can coexist. This is the approach of `Poetry`. In this case, it's one global lockfile that's used for the individual groups and for the combinations.  
  
The PEP doesn't enforce anything with that regard for tools, it's totally up to them to do whatever they want.



##### This PEP and tools



This section will serve as a base of what we can expect from tools in terms of functionality and behaviour. It's the only PEP in the ones I've covered in this article that details really well its interaction with tools. So I thought it's educational and interesting to lay out what's scattered throughout the PEP about how should tools behave, when they should emit errors, what the PEP doesn't expect from them and is totally up to the tools etc.   
  
Obviously, tools aren't supposed to respect a PEP exactly as it is and they might have their own opinions on certain things. Especially in the early days of a PEP's acceptance. So it's not because a PEP advises tools to do or not to do some things that it's how the tools will act. Keep that in mind and always try the tools before assuming anything.  
  
First, I have to say that this PEP doesn't get into how tools should install or manage dependency groups. The implementation details and CLI interface is totally left to the tools, which is reasonable. But, it has some suggestions for **build backends** specifically.  
  
Let's start with what tools **should do**:  
- Tools should present non normalized group names to users. In the `pyproject.toml` dependency groups might contain hyphens, uppercase letters etc. Tools should present non normalized names to users. They do have to normalize the names internally as part of the groups expansion and resolution etc., but when presenting that data to users, it should be kept as in the `pyproject.toml`.  
- Tools should ensure that the non normalized group names are [valid non normalized names](https://packaging.python.org/en/latest/specifications/name-normalization/#valid-non-normalized-names), and handle the [normalization](https://packaging.python.org/en/latest/specifications/name-normalization/#normalization) correctly.  
- Tools should process Dependency Group Includes by expanding them exactly at the location of the include without altering the sequence  
- Tools should handle the resolution strategy for dependency groups as they would do in any other case. For example if a group's list of dependencies contains the same dependency with different version constraints, tools should handle that case as they would do normally. And when this case arises due to inclusion through Dependency Group Includes, they still should handle that case as they would do in any other case. Whether find a version that satisfies all constraints or report an error if they're mutually exclusive etc. (so tools shouldn't have some kind of weird behavior like omit the "faulty duplicate" dependency from the include).  
- Obviously tools that support Dependency Groups should provide an interface for installing from Dependency Groups. **BUT**, tools may choose to provide the same interface as they do for installing `extras`.  
- And obviously should document the usage, and they should document the issues you might run into, like `uv` [informs](https://docs.astral.sh/uv/concepts/dependencies/#dependency-groups) that if you have conflicting groups in your dependency groups, then it'll report an error (see <https://github.com/astral-sh/uv/issues/6981>). It doesn't allow that yet. The same goes for `extras` or if you have a conflicting group and extra. The interface between `extras` and `optional` and dependency group might be a little bit confusing in the beginning for `uv`.  
- This is goes hand in hand with the last point of what tools should not do, tools should only validate Dependency Groups they're using.  
- Tools should do as they see fit (since it's out of scope for this PEP) with mutual compatibility of global compatibility of dependency groups when trying to install in the same environment different dependency groups (important to remember that so you don't get confused with the eager validation recommendations).  
- For environment managers (like `tox`, `nox` or `hatch`), they should support (reading, adding, using) dependency groups in the same way they do with the dependencies declared in their configuration files. This is to centralize dependency management in `pyproject.toml` and reduce duplication and potential inconsistencies between configurations.   
  
What tools **should not do**:  
- Tools should not deduplicate or otherwise alter the list contents produced by the include. It's supposed that the Dependency Groups data, whether as a list of dependency specifiers or using Dependency Groups Includes, is truthful and absolute data, and when one group includes another, the contents of the included group are inserted into the current group at the point of inclusion. Tools should not modify the resulting list of dependencies after inclusion, they should keep things as intended by the user. On top of that, there's another good reason for keeping things as they're, it's the resolution behavior, changing the orders of dependencies can change what versions of sub-dependencies are installed.  
- Tools should not eagerly validate the dependency groups. Which means that tools should avoid validating Dependency Groups that they are not currently using. This is so good because it reduces unnecessary errors or warnings for unused groups, sometimes you know you are not going to use a group of dependencies ever but you just have it here just in case. And that would allows different tools or processes to use different Dependency Groups without interference. This joins the fifth point of when it's preferable for tools not to report an error. Imagine you have a `tool1` that supports features up to a `PEP-X`, you want to use it with dependency group `group1`, and you also have a `tool2` that supports up to `PEP-Y` where `PEP-Y` was accepted much later than `PEP-X` and you want to use this tool with `group2` because of some nice new features. `tool1` should not prevent you from doing that just because it sees some weird data according to it. This suggestion is mainly for tools that install or resolve dependency groups, so tools that impact the dependency graph, but the suggestion doesn't hold for linters and validation tools because these tools check code or configuration files for errors, best practices, or policy compliance so they might need to validate all Dependency Groups to ensure the overall integrity of the `pyproject.toml` file.  
  
When should tools **emit an error** and when **it's preferable if they don't**:  
- If after normalization of group names duplicates are found, tools should emit an error. No two groups with the same normalized names are allowed to coexist.  
- Tools must emit an error if a Dependency Group Includes contains a cycle.  
- This is not a compulsory error emission requirement but tools may choose to report an error when a user provides the same name for a dependency group and for an optional extra. The PEP advises users not to do so by the way.  
- Tools should emit an error though if there are duplicate dependency group names after normalization.  
- This is also not a compulsory error emission requirement, but since this PEP might be extended in the future with the introduction of new data, tools should be careful not to perform a rigid data validation check when they validate either the `dependency-groups` table or the whole `pyproject.toml` file. They should only ensure that `dependency-groups` are correctly declared up to the specification / PEP the tools support, and leave room for users to use others tools that support more advanced PEPs. So a good approach would be to only validate known fields and tables according to the tool (to the standards it implements) and ignore unknown fields and tables without error (obviously the fields and tables should be `dependency-groups`, this PEP only talks about that, if there are some unknown fields in the `project` table, then the tool should check what the corresponding PEP suggests). So tools might choose to report an error here, but it's better if they don't.  
- Obviously tools must emit an error when they encounter weird data in `dependency-groups` but with respect to the PEP they support. Like have a Dependency Group Includes that is not a table, or that uses a colon`:`instead of the equal sign `=` etc.  
  
What tools **must do**:  
- For build backends specifically, support for Dependency Groups will require support for inclusion from the `project` table. From my understanding, if build backends allow inclusion of dependency groups in the `project` table (one way or another), they should support this inclusion appropriately. By the way this interaction between `dependency-groups` and `project` only happens for build backend. This is just something to **future-proof the PEP**. Build backends might want to use dependency groups internally, either when evaluating dynamic metadata or when resolving dependencies during the build process etc. So if they do that, **if they chose to use/include the dependency groups in the `project` table in one way or another, then they should support that appropriately**. It's **not something that this PEP defines**. It **might be an area for future standardization**. This PEP does not specify how they should do this.  
- Tools must ensure that Dependency Group Includes are acyclic to prevent infinite loops or recursion.   
  
What tools **must no do**:  
- Build backends must not include Dependency Group data in built distributions as package metadata. When building a package, the `dependency-groups` data defined in `pyproject.toml` should not be included in the package's metadata (e.g., `PKG-INFO` or `METADATA` files). That defeats their purpose.



We have covered pretty much the whole [PEP 735](https://peps.python.org/pep-0735/)! I still advise you to read it, or at least the [rejected ideas](https://peps.python.org/pep-0735/#rejected-ideas), [deferred ideas](https://peps.python.org/pep-0735/#deferred-ideas) and the [implementations in the other programming languages](https://peps.python.org/pep-0735/#appendix-a-prior-art-in-non-python-languages).



#### All the ways for declaring sources with uv



There are various ways in `uv` to specify the source from which to get a dependency, these are not covered by any PEP or specification in my knowledge. This is also a feature that many tools provide, not only `uv`.  
  
In `uv` there are five different types of sources for [dependencies](https://docs.astral.sh/uv/concepts/dependencies/): `git`, `URL`, `path`, `workspace` and `index`.



- **Git repository**:
  - This is to add a dependency directly from a git repository.
  - To add from git you do `uv add git+https://github.com/user/repo`
  - You can add from a tag, which allows you to pin a dependency to a specific release, with `--tag <TAG>`.
  - You can add from a specific with `--branch <BRANCH>`.
  - You can add from a specific commit hash with `--rev <REV>` (rev means **revision**), this is for example when you want to ensure reproducibility.
  - All of these can be added manually to the `pyproject.toml` in the `tool.uv.sources` table where keys are your dependencies and the values are tables that look like `{ git = ..., tag = ...}`.
  - You can add from a subdirectory of the repository and if the package isn't in the root directory you can add `subdirectory` manually in the `pyproject.toml`: `{ git = ..., subdirectory = ...}`.
- **Direct URL**:
  - This is to add dependencies from a URL pointing to a wheel `.whl` or a source distribution `.tar.gz` or `.zip`.
  - You just do `uv add <URL>`.
  - This looks like `{ url = ...}`. With the option to add a subdirectory if the package isn't in the archive root as well: `{ url = ..., subdirectory = ...}`.
- **Local Path**:
  - This is to add a dependency from a local file path. It can be a wheel, a source distribution, or simply project directory.
  - You just do `uv add /local/path`
  - But it offers the option to install the dependency in editable mode with `--editable` so that whenever you make changes in the dependency, they're immediately provided.
  - This looks like `{ path = ... }` or `{ path = ..., editable = True }`.
- **Specific Index**:
  - This is to specify from which index certain dependencies have to be resolved from.
  - There is a `--default-index` option to override the default index from which to resolve the package. It's the index that takes the lower priority when resolving from different indices though.
  - And there is a `--index` to specify an index from which to resolve a package. You can specify many indices in one command `uv add package --index index_1 --index index_2`. The earliest takes priority.
  - Example: `uv add httpx --default-index https://anaconda.org/conda-forge/httpx --index https://pypi.org/simple`. This leads to the following `pyproject.toml`, which you can also get manually:  
    `[project]  
    ...  
    dependencies = [ "httpx>=0.27.2" ]  
      
    [[tool.uv.index]]  
    url = "https://anaconda.org/conda-forge/httpx"  
    default = true  
      
    [[tool.uv.index]]  
    url = "https://pypi.org/simple"`
  - If no index is specified, then the default one is PyPI. So, **PLEASE USE CACHING** TO NOT OVER DOWNLOAD FROM PYPI. `uv` does that by default but I wanted to seize the opportunity to say it. For `uv` you can find that data in `$XDG_CACHE_HOME/uv or $HOME/.cache/uv` on macOS and Linux, or `%LOCALAPPDATA%\uv\cache` on Windows. For example, I can do that `ls $HOME/.cache/uv/wheels-v2/pypi/httpx/httpx-0.27.2-py3-none-any/httpx` to find the cached `httpx` from the `pypi` index.
- **Workspace Member**:
  - A workspace member is just another package in the same workspace your project is in.
  - If all of the above allowed to add dependencies with the CLI, workspace members require manual intervention.  
    `[project]   
    dependencies = [  "other_package == 0.1.0" ]    
      
    [tool.uv.sources]   
    other_package = { workspace = true }    
      
    [tool.uv.workspace]   
    members = [  "packages/other_package" ]`
  - We'll do a deep dive on workspaces in the second part of this article, there is cool stuff you can do like exclude some members etc.



All of the above can be removed by `uv remove` directly, with no source declaration. The souces won't be necessarily removed from the `pyproject.toml` though, I think the path sources are removed but index sources are not for example.



## Interaction with lockfile and virtual environment



I'm not sure that the vocabulary of this section is shared by all tools, but I'm sure a good part of them does, and a bigger part has the same workflow for doing things which is to go from `pyproject.toml` to the lockfile (or an equivalent) to the virtual environment. I think understanding the interaction with these components is important for managing your projects.  
  
Let's start with understanding some basic notions.  
  
**What is re-locking?** We have already talked about lockfiles, they're files that record the exact versions of dependencies and their dependencies after the tool's dependency resolution. Re-locking is updating the lockfile (in our case, the `uv.lock`) to match the current dependencies in the `pyproject.toml`. It's important to ensure that the lockfile accurately reflects the exact versions of all dependencies for reproducibility. As we said, it's an important file that we push to git.  
  
**What is syncing?** Syncing is making sure that the virtual environment reflects the exact package versions specified in the lockfile. It's important to ensure consistency between the project's resolved dependencies and the actual environment.  
  
So **we go from the `pyproject.toml` where we declare dependencies, to the lockfile that records the dependency resolution, to the virtual environment where the result of the resolution is installed**.  
  
Commands like `uv add`, `uv remove`, and `uv run` will generally act on the lockfile to either ensure it matches the dependencies declared in the `pyproject.toml`, or inline dependencies with scripts if you're using `uv run` on scripts, but that's for part 2 (we'll see more on `uv run` in that part as well). They'll also act on the virtual environment to sync it to the new resolved dependencies.  
  
Now, these three commands will provide the options `--locked`, `--no-sync` and `--frozen`. These options will decide whether the lockfile or the virtual environment or both are changed or not, BUT, the **`pyproject.toml` is always updated**.  
  
The `--locked` option disables the lockfile update. No dependency resolution is performed but the command won't run if the lockfile is not up to date! And the command does update the virtual environment!  
It's generally used in continuous integration or deployment scenarios **where you want to ensure consistency and prevent any changes to the lockfile**.  
  
The `--no-sync` option disables the update of the virtual environment. It updates the lockfile though, and since it updates it, it does perform dependency resolution. Generally you do that when you have a lot of dependencies or big dependencies, you want to update the lockfile to match the `pyproject.toml` but don't want to install them at the moment for one reason or another (you just want to have an up to date lockfile and push that to Git or you need the up to date lockfile but don't need the dependencies themselves in your environment because what you're working on only uses a subset of the dependencies etc.)  
  
The `--fro`zen option disables updating both the lockfile and the virtual environment. And so obviously it performs no dependency resolution, and since it updates your `pyproject.toml`, that means that **if you provide no bounds for your dependencies, then they'll be added to the `pyproject.toml` with no bounds either**, as opposed to the default behavior.



# Some Practical uv Workflows



Some typical things I do with `uv`:  
  
- Clone a data science repository with a `requirements.txt` file and I only want to run a file: `uv run --with-requirements requirements.txt main.py`  
Since it's too long to do, I create a shell function to do that, nothing fancy but honest work  
`uvreq() {  
 uv run --with-requirements "$1" "$2"  
}`  
  
- If I want to maintain a project with a `requirements.txt` file then I clone the project, I `uv init` to get a `pyproject.toml`, then `uv add -r requirements.txt` to get a lockfile and the virtual environment up and running from the `requirements.txt`. Put both the `pyproject.toml` and the `uv.lock` in the `.gitignore`. If I have to add or remove some packages, I use the `uv` commands as if I'm working on a package project, but then before I push the code, I `uv export`.  
`uv` offers a great deal of flexibility where you can play with extras or dependency groups while keeping the `requirements.txt` and only exporting to if deemed necessary.  
  
- If I have `requirements.txt` for a project but those requirements are only for some specific group of dependencies and I don't have the project's dependencies or don't want to bother with it, I can install the `requirements.txt` as a dependency group with `uv add --group reqs -r requirements.txt`. If you don't want to create a virtual environment for the moment because you have different requirements for different groups and you just want to add them to the `pyproject.toml` then you can do `uv add --group reqs -r requirements.txt --frozen`.



# Thanks



I thank all the readers and commenters on Reddit that took the time to point out issues with my website or with my article.



# EDITS



- In the introduction: "There are many other similar tools such as [Poetry](https://github.com/python-poetry/poetry), [PDM](https://github.com/pdm-project/pdm), [tox](https://github.com/tox-dev/tox) etc. Don't hesitate to explore and try them and pick which suit your needs. [pip](https://github.com/pypa/pip) might be sufficient by itself for your needs. Again, the goal of this article is not to be a guide of a particular tool, but to get you from knowing nothing, to knowing the standard."  
User u/-defron- [commented](https://www.reddit.com/r/Python/comments/1gkmrsg/comment/lvnarac/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) that it was confusing to mention `tox` among `Poetry` and `PDM`, which is right.



- In the high-level overview of "Dependencies": "There 5 types of dependencies that you should know about when it comes to Python projects. The normal project dependencies, the optional dependencies, the development dependencies and dependency groups, and finally the editable dependencies." changed to: "There 4 types of dependencies that you should know about when it comes to Python projects. The normal project dependencies, the optional dependencies, the development dependencies and dependency groups, and finally the editable dependencies."  
And still in the high-level overview of "Dependencies", I merged the list for editable dependencies in the normal dependencies of a project. The old version of the impacted elements was:



- Project **dependencies**:
  - They are just what you're used to, the essential packages that your project requires to function properly.
  - Inclusion in `pyproject.toml`: as a [PEP 508](https://peps.python.org/pep-0508/) compliant string in the `dependencies` key (which is an array of these strings) in the `project` table.
  - How to add them using `uv`: `uv add <package> ...`
  - Example: `uv add numpy pandas`
  - Standard: [PEP 621](https://peps.python.org/pep-0621/)
- **Editable dependencies**:
  - These are packages that are under development, they can be local packages or not. They're still dependencies so they must be added to `dependencies`.
  - We separate them out just to point out that you can get the changes made to these dependencies instantaneously in your project.
  - Standard: like development dependencies, there is no standard way to specify them. I'm talking about a standard way to define them in `pyproject.toml` and not some kind of standard for editable dependencies like is talked about in [PEP 660](https://peps.python.org/pep-0660/).
  - How to add them using uv: `uv add --editable <package> ...` where package can be the path to your local package, or a git repo etc.
  - `uv` will add the package to `dependencies` and will also add the **source** (the path, the git repo etc.) to the table `tool.uv.sources` as a key with the same name as your package and whose value is a mapping that contains the source.



The new version doesn't contain the list on editable dependencies but adds it as an element to the project dependencies: "Non-included in the standard: A particular case of project dependencies are **editable dependencies**. These are packages that are under development, and that you want to include in an **editable** mode, which is a mode that will reflect any changes made to these dependencies immediately in your project. You can add them by using uv: `uv add --editable <package> ...` where `package` is the path on your filesystem to the dependency. `uv` will add the package to `dependencies` and will also add this `<package-name> = { path = <package-path>, editable = True}` in the `tool.uv.sources` table `tool.uv.sources`. And as pointed out by this [comment](https://www.reddit.com/r/Python/comments/1gkmrsg/comment/lvofx3m/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) (from u/ArchFFY00), there is no disparity in this feature between the different tools and this (the inclusion of editable dependencies) relates to the lockfiles which are not covered by the standard. As mentioned, an editable dependency *is* a dependency, so it's included in the `dependencies` of a project, it's just *how* the tool includes that dependency as an editable one that will differ from one tool to another.



I have also corrected the typo that we can add an editable from a different source than the filesystem. Maybe it can work if it's a local git repo and you add it as an editable instead of adding the package itself as editable. But I haven't tested. And made the paragraph nicer (I believe).  
  
This was thanks to the [comment](https://www.reddit.com/r/Python/comments/1gkmrsg/comment/lvofx3m/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) of u/ArchFFY00 who pointed out that editable dependencies are not really some type of dependency as are optional dependencies. Which is totally right. He also pointed out that there is no disparity between the tools when it comes to editable dependencies and how that relates to the lock file which is not in the standard. And that it's confusing to have them listed there as a different kind of dependencies, which is totally right as well.



- I also changed this small note: "**Note**: there are also build dependencies, but that's not really the scope of these commands." to "Note: there are also build dependencies, but that's not really the scope of this part.".

---

<div class="comments-section">
<h2 class="comments-title">15 comments (archived)</h2>
<p class="comments-note">These comments were migrated from the original WordPress site. New comments are via Giscus below.</p>
<div class="comment" id="comment-178" style="margin-left:0em">
  <div class="comment-meta">
    <span class="comment-author">John</span>
    <span class="comment-date">November 12, 2024</span>
  </div>
  <div class="comment-body"><p>Really good article. I&#x27;m looking forward to the next one in this series. That said, there are a number of grammar issues with this that I hope can be improved.</p></div>
  <div class="comment" id="comment-179" style="margin-left:2em">
    <div class="comment-meta">
      <span class="comment-author">admin</span>
      <span class="comment-date">November 12, 2024</span>
    </div>
    <div class="comment-body"><p>Hi John!</p><p>Thanks a lot for reading and for the comment. It&#x27;s encouraging to hear.</p><p>I&#x27;ll aim to reduce grammar issues as much as possible.</p><p>Some comments on Reddit also suggested making the content more easily consumable, so I’m planning to restructure this article into several smaller pieces while improving the grammar along the way. I’ll keep this longer version on the blog though. I’ll do the same for the next part: posting a lengthy article, then dividing it into smaller, more easy to read articles.</p></div>
  </div>
</div>

<div class="comment" id="comment-180" style="margin-left:0em">
  <div class="comment-meta">
    <span class="comment-author">Mathieu</span>
    <span class="comment-date">November 12, 2024</span>
  </div>
  <div class="comment-body"><p>Very interesting article, thanks a lot!</p></div>
  <div class="comment" id="comment-181" style="margin-left:2em">
    <div class="comment-meta">
      <span class="comment-author">admin</span>
      <span class="comment-date">November 12, 2024</span>
    </div>
    <div class="comment-body"><p>Thank you! That encourages me to continue!</p></div>
  </div>
</div>

<div class="comment" id="comment-182" style="margin-left:0em">
  <div class="comment-meta">
    <span class="comment-author">Etienne</span>
    <span class="comment-date">November 12, 2024</span>
  </div>
  <div class="comment-body"><p>Very very nice article, lots of great information.</p></div>
  <div class="comment" id="comment-183" style="margin-left:2em">
    <div class="comment-meta">
      <span class="comment-author">admin</span>
      <span class="comment-date">November 12, 2024</span>
    </div>
    <div class="comment-body"><p>Thanks Etienne! I&#x27;ll aim to make the next part as comprehensive as this one and cover the remaining steps to complete the Python packaging workflow from initializing a project to building it and publishing.</p></div>
  </div>
</div>

<div class="comment" id="comment-184" style="margin-left:0em">
  <div class="comment-meta">
    <span class="comment-author">Abed</span>
    <span class="comment-date">November 15, 2024</span>
  </div>
  <div class="comment-body"><p>Thanks, Etienne. This is an excellent article. I look forward to the next part, which hopefully covers publishing into a private index repository (e.g., gemfury). Also, I am wondering if UV works well with tools like Pyarmor or if there are plans to integrate such functionality. Final question: Is there an equivalent pip.ini file that allows me to store the --default-index and --index?</p><p>Thanks again.</p></div>
  <div class="comment" id="comment-185" style="margin-left:2em">
    <div class="comment-meta">
      <span class="comment-author">admin</span>
      <span class="comment-date">November 17, 2024</span>
    </div>
    <div class="comment-body"><p>Hey Abed, thank you a lot for your comment!</p><p><br>For your first question, no need to wait for the second part! You can do `uv publish --publish-url &lt;private-url&gt; --username &lt;your-username&gt; --password &lt;your-password&gt;` or `uv publish --publish-url &lt;private-url&gt; --token &lt;token&gt;`. Or you can use environment variables, `UV_PUBLISH_URL`, `UV_PUBLISH_USERNAME`, `UV_PUBLISH_PASSWORD`, `UV_PUBLISH_TOKEN`.<br>You can also add `publish-url` to your `pyproject.toml` or `uv.toml`.</p><p><br>For the question on `pip.ini` and default index, I guess you&#x27;re talking about downloading right? You can use the `uv.toml` file as an equivalent to `pip.ini`.<br>If you want to set up a default index for all your packages and don&#x27;t want this metadata leaked to the public somehow, you can use a `uv.toml` and do this:<br>````<br>[[index]]<br>url = your-default-index<br>default = true</p><p>[[index]]<br>url = fallback-index<br>```<br>You can also use the CLI `uv add &lt;package-name&gt; --default-index &lt;default-index&gt; --index &lt;fall-back-index&gt;` and that will update your `pyproject.toml` with the relevant metadata. <br>In both cases it&#x27;s not necessary to have that fallback index. And in case you add a default index per package, that default won&#x27;t be the same for other packages (they&#x27;ll be defaulted to PyPI) obviously.<br>You can also use environment variables `UV_DEFAULT_INDEX` and `UV_INDEX_URL`.</p><p><br>Personally I don&#x27;t like environment variables in this case that much, besides the security risks they&#x27;re global and aren&#x27;t flexible. And depending on the security constraints I have and in what environment I&#x27;m operating in, I wouldn&#x27;t want to provide my token or my password in the CLI as well (access to shell history or process list etc.). <br>Unfortunately for the moment only `uv run` can consume `env` dotfiles like `.env` etc.</p><p><br>I&#x27;m smoothly writing the next part, and guess what, I just finished writing a quick example about using `pyarmor` as a thank you for your comment. I&#x27;m not sure about how you specifically used pyarmor or want to use it, but I hope the toy example as well as the article will facilitate for you integrating pyarmor in your projects. I haven&#x27;t yet figured out a seamless way for integrating it, but I believe the article will equip anyone, even a beginner, with the right knowledge to integrate any tool.</p><p><br>About uv and pyarmor:</p><p>- So if you&#x27;re talking about uv integrating pyarmor as part the build process when building a package, it&#x27;s not the case right now. uv doesn&#x27;t have its own build backend yet and they&#x27;re working on it (https://github.com/astral-sh/uv/issues/3957). But, you can use a different build backend like hatchling and you can integrate pyarmor into your build process through that. And I can&#x27;t know about whether uv&#x27;s plans about integrating such tools since they didn&#x27;t communicate about that (I believe).</p><p>- If you&#x27;re talking about just using pyarmor as a CLI tool with uv, then you can do stuff like `uv run pyarmor gen foo.py`, if you have pyarmor as part of your dependencies or dev dependencies etc. You can also do `uv run foo.py` while having pyarmor as inline dependency in the foo.py script. I might cover inline dependencies in a bonus part, along workspaces and some other stuff, but there isn&#x27;t much to them.</p><p><br>I&#x27;m explaining all of why you can use almost whatever build backend you want with tools like uv, in the next article. And why it&#x27;s such a great thing. It was such a pleasure reading through PEP 517 and seeing how they isolate the build frontend from the build backend.</p><p></p><p>(I&#x27;m not Etienne by the way but it&#x27;s okay!)</p></div>
  </div>
</div>

<div class="comment" id="comment-190" style="margin-left:0em">
  <div class="comment-meta">
    <span class="comment-author">Abed</span>
    <span class="comment-date">November 21, 2024</span>
  </div>
  <div class="comment-body"><p>What a treat, thank you. I am looking forward to go over the article and lots of learning. Many thanks.</p></div>
  <div class="comment" id="comment-191" style="margin-left:2em">
    <div class="comment-meta">
      <span class="comment-author">admin</span>
      <span class="comment-date">November 22, 2024</span>
    </div>
    <div class="comment-body"><p>You&#x27;re welcome! I hope that helps you with whatever you&#x27;re doing :D</p></div>
  </div>
</div>

<div class="comment" id="comment-241" style="margin-left:0em">
  <div class="comment-meta">
    <span class="comment-author">Abed</span>
    <span class="comment-date">February 16, 2025</span>
  </div>
  <div class="comment-body"><p>Hello, thank you for this resource, it has been of great use for me. Many thanks. One question that I would appreciate your help with, is how can I integrate uv into a windows installer like InnoSetup, to first install uv, and then use uv to install tools from pypi and private index. Is that possible? Any hints on how to start (or if you recommend another tool than InnoSetup) would be greatly appreciated. KR, Abed</p></div>
  <div class="comment" id="comment-251" style="margin-left:2em">
    <div class="comment-meta">
      <span class="comment-author">admin</span>
      <span class="comment-date">February 21, 2025</span>
    </div>
    <div class="comment-body"><p>Hi Abed, sorry for the late response.</p><p>Unfortunately I&#x27;m not familiar with InnoSetup and installers on Windows in general so I can&#x27;t really help here. </p><p>I think you should ask on their forum if they have one or ask on uv&#x27;s repo on GitHub.</p><p>Good luck!</p><p>ReinforcedKnowledge</p></div>
  </div>
</div>

</div>
