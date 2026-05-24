+++
title    = "Python Project Management and Packaging: PEP 751 update and some of the remaining issues of packaging"
date     = "2025-05-02T06:50:50+00:00"
draft    = false
categories = ["Transformers"]
+++

My first two articles ([part 1](https://reinforcedknowledge.com/a-comprehensive-guide-to-python-project-management-and-packaging-concepts-illustrated-with-uv-part-i/) and [part 2](https://reinforcedknowledge.com/a-comprehensive-guide-to-python-project-management-and-packaging-concepts-illustrated-with-uv-part-2/)) on Python project management and packaging gathered a lot of interest and I thought they were comprehensive enough for me not to come back and update them for a moment.   
  
But, about a month ago, [PEP 751 – A file format to record Python dependencies for installation reproducibility](https://peps.python.org/pep-0751/), was accepted, 31 March 2025. At the time of publishing my first two articles, I didn't think that lockfiles would get standardised and I didn't know about the ongoing effort.  
  
So I thought of writing this article, both as an update to my previous two articles but also to talk about some issues that still remain with Python packaging and that would require more fundamental work in how we do things. What was said in the first two articles about lockfiles is still correct. But at the time of writing there was no approved standard for them, and this update will talk about it.



# Update: lockfiles



PEP and specification: [PEP 751 – A file format to record Python dependencies for installation reproducibility](https://peps.python.org/pep-0751/#file-format)



## Why do we need lockfiles



If you are not familiar with lockfiles, you're probably familiar with the `requirements.txt` file. If you are not familiar with it, in its simplest form, it's a text file that's used by people to either record the dependencies of their project or their project's environment. There is a difference. Recording the dependencies of the project means you're communicating what your project depends on directly. Recording the project's environment means, in its simplest form again, recording even the transitive dependencies among other things. That might not be enough, you might require some information to completely record the environment, and we're going to find out what it is as well.  
  
Let's say you are starting a data science project and you depend on both `pandas` and `numpy`. You can write the following `requirements.txt` file. The name `requirements.txt` is just a convention, you can use whatever name you want.



```
numpy
pandas
```



That obviously is bad, you should at least put some constraints on the versions of your packages otherwise you can find yourself with a bunch of issues due to your project dependencies changing every time you're reinstalling them, your coworkers might not be able to install the same packages because they're on a different platform or using a different Python version etc. So, you might opt for something like:



```
numpy==2.2.4
pandas==2.2.3
```



This is better, but it still suffers from a lot of issues. You're now recording exactly what dependencies your project depends on, but not at all what your environment is. So a colleague of yours might develop the same package on a different environment. This is bad. Why? Because the file, as it is right now, says nothing about transitive dependencies. Transitive dependencies are the dependencies that your dependencies depend on. Example: [`pandas` depends on `python-dateutil`](https://github.com/pandas-dev/pandas/blob/c27a309854401cc1931cd72ba1c5c99e77472fe4/pyproject.toml#L32). The dependency is specified as `python-dateutil>=2.8.2`. What would happen when `python-dateutil>=2.8.3` is released and you, or in the CI, or a new dev on the project does `pip install -r requirements.txt`? Well, it's the new version of `python-dateutil` that will be installed instead the one you, the old dev, have in your `.venv`. There are other cases that might cause the same issue, imagine two developers on different operating systems. The author releases version 2.8.3 of `python-dateutil` wheels only for Windows. The Windows user immediately gets the wheel, while the Linux user gets exactly the same 2.8.3 code but as a source distribution and has to build it from it. So it's not because we have pinned a version that everyone will get the exact wheel from it.  
  
This is even worse if you have to build from an sdist, whether because the dependency or its version is too new and there is only an sdist yet for your platform or whether you're using a platform the authors of the dependency didn't build a wheel for it yet. And this is worse because there are so many things that are beyond the scope of `requirements.txt`. Everything that's required to build the sdist into a wheel for your platform are beyond the scope of the project, so the build backend, the `build-requires`, whatever dynamic stuff that potential build hooks will do, the compiler toolchain of the OS, the compiler flags, the environment variables on your machine etc. All of that is beyond the scope of what `requirements.txt` can pin down. And it's easy to see how this can lead to building different wheels on different platforms for a dependency that requires to be built from an sdist. `requirements.txt` can reference an sdist or VCS URL but it cannot express how the build must be carried out, so you have no control on the outcome.  
   
Let's first fix the first issue and include the transitive dependencies, we can do that with `pip freeze > requirements.txt`:



```
numpy==2.2.4
pandas==2.2.3
python-dateutil==2.9.0.post0
pytz==2025.2
six==1.17.0
tzdata==2025.2
```



You'd think we're out of trouble with respect to the first issue. But no, not only you are not 100% sure that you can reproduce the exact environment, but you also have security issues, especially against supply chain attacks. You are not 100% guaranteed to reproduce the exact environment because even if you pin `numpy==2.2.4`, the maintainers can upload extra wheels for that same version (for example, a build for a newly released Python version). Those extra wheels have new hashes, so the exact bits you download today might differ from the ones you downloaded yesterday. And with a similar mechanism, you're vulnerable to supply chain attacks, malicious code can infiltrate its way into one of your dependencies while keeping the same exact version number. So you have frozen records or versions, but not the files. You can think of a snapshot of what the environment looks like but not really a lock of that one.



Now come the hashes. Hashes allow for file immutability. Pinning a, say, SHA‑256 digest ties the requirement to one specific file and anything else is rejected. But you should understand what these offer so that you can judge the security guarantees their offer. A hash only tells you whether the file changed or not since you calculated the hash for it. It doesn't guarantee the authenticity of the initial file. That'd be the role of a signature.  
  
What we need to do now is turn our `requirements.txt` above in a `requirements.in` file. Then install `pip-tools` and do `pip-compile requirements.in --generate-hashes > requirements.txt`. That will generate a `requirements.txt` where every package, including transitive requirements, appear with one or many hashes (if there are many wheels, like for different platforms):



```
#
# This file is autogenerated by pip-compile with Python 3.12
# by the following command:
#
#    pip-compile --generate-hashes requirements.in
#
numpy==2.2.4 \
    --hash=sha256:05c076d531e9998e7e694c36e8b349969c56eadd2cdcd07242958489d79a7286 \
    --hash=sha256:0d54974f9cf14acf49c60f0f7f4084b6579d24d439453d5fc5805d46a165b542 \
    --hash=sha256:11c43995255eb4127115956495f43e9343736edb7fcdb0d973defd9de14cd84f \
    ...
    # via
    #   -r requirements.in
    #   pandas
pandas==2.2.3 \
    --hash=sha256:062309c1b9ea12a50e8ce661145c6aab431b1e99530d3cd60640e255778bd43a \
    --hash=sha256:15c0e1e02e93116177d29ff83e8b1619c93ddc9c49083f237d4312337a61165d \
    --hash=sha256:1948ddde24197a0f7add2bdc4ca83bf2b1ef84a1bc8ccffd95eda17fd836ecb5 \
    ...
    # via -r requirements.in
python-dateutil==2.9.0.post0 \
    --hash=sha256:37dd54208da7e1cd875388217d5e00ebd4179249f90fb72437e91a35459a0ad3 \
    --hash=sha256:a8b2bc7bffae282281c8140a97d3aa9c14da0b136dfe83f850eea9a5f7470427
    # via
    #   -r requirements.in
    #   pandas
pytz==2025.2 \
    --hash=sha256:360b9e3dbb49a209c21ad61809c7fb453643e048b38924c765813546746e81c3 \
    --hash=sha256:5ddf76296dd8c44c26eb8f4b6f35488f3ccbf6fbbd7adee0b7262d43f0ec2f00
    # via
    #   -r requirements.in
    #   pandas
six==1.17.0 \
    --hash=sha256:4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274 \
    --hash=sha256:ff70335d468e7eb6ec65b95b99d3a2836546063f63acc5171de367e834932a81
    # via
    #   -r requirements.in
    #   python-dateutil
tzdata==2025.2 \
    --hash=sha256:1a403fada01ff9221ca8044d701868fa132215d84beb92242d9acd2147f667a8 \
    --hash=sha256:b60a638fcc0daffadf82fe0f57e53d06bdec2f36c4df66280ae79bce6bd6f2b9
    # via
    #   -r requirements.in
    #   pandas
```



Now, to get `pip` to insist that every package matches a recorded hash you need to add an extra flag during the installation: `pip install --require-hashes -r requirements.txt`. `pip`'s default behaviour is to abort the installation of a package it couldn't verify the hash against the recorded ones and now we ask it to require a hash for every package. Notice how every line has both the hash algorithm and the hash value. It's important so that pip's resolver can verify whether the hash of the new package it fetched matches or not the hash mentioned in the `requirements.txt` file.



So for now with this freeze+hashes workflow we're able to have reproduce the *current* environment, at least when working with wheels (let's forget about sdists for a moment). But the reproducibility we have achieved here is of the current environment alone. Our hash‑pinned `requirements.txt` records the exact environment that was used to produce it, so same OS, same CPU architecture, same Python implementation and version, same package index layout etc. It does not automatically give you file that works on other platforms or Python versions etc.  
  
The other issues we'd like to solve is how to *easily* include extras or dependency groups (we can generate a different file for each one of them), how to trace where the package should be looked for and how to authenticate it (index, its identity etc.). How to generalize to other platforms (for the moment if a package has a Windows-only transitive dependency but we're generating our requirements file on a Linux distribution then we won't capture that) etc. A bunch of issues that would require from us a lot of workarounds and complex worklows and communication so that our users/contributors follow them exactly.



lockfiles' goals are to make all of this easier and improve on the cross-platform part, the security guarantees etc.



## pylock.toml



PEP 751 standardizes a first version of lock files and says that the filename(s) *must* either be `pylock.toml` or `pylock.*.toml`. Ideally, `pylock.<service or environment or etc., something semantically meaningful>.toml` like `pylock.dev.toml`. The prefix and suffixe of the filename *must* be lowercase. Services or tools will look for `pylock.<service name or tool name>.toml` in case you have multiple *single-use* lock files, and if you have one *multi-use* lock file, `pylock.toml`, then the service or tool will look for the appropriate dependency group in it.  
  
The lock file *should* be placed in the same scope as where the environment it describes lives. E.g., in the same scope as the `pyproject.toml` against which it was generated. For monorepos with different projects (we'll talk more about that later on), it should be placed at the root of the repo.   
  
From here on when talking about the lock file, I'll refer to the standard's, and call it `pylock.toml` for simplicity. If I'm going to talk about a different lock file format, I'll refer to it with its name (e.g., `uv.lock`).



**What is a lockfile in general?** a lockfile is an *immutable* file that records *completely* and *exactly* a given environment.   
  
You can think of the final file we produced above as some kind of lockfile as well.   
  
Most lockfiles are `toml` files, which makes them easy to be read and audited by humans.  
  
It's important to understand what we mean by reproducibility. Do we mean the capacity to reproduce the current environment given the same platform at any given time, or the capacity to reproduce the current environment across different platforms and at any given time. Lockfiles that allow only the former are usually referred to as single-use lockfiles. Lockfiles that extend to the latter are usually referred to as multi-use lockfiles. A platform is not only defined by the OS, the Python implementation / version, CPU architecture etc., but also the extras / dependency groups.



Due to the importance of lockfiles and the information they control, they must be included in your version control. And a lot of work was done to make them work well when comparing diffs, this should be appreciated.



Let's understand of what is composed a `pylock.toml`, so that later on we can understand how it solves what we talked about before and what new things it introduces and that we couldn't do.  
  
We first have the top level keys:



- `lock-version`: it's a **required** key, and it indicates the file format version of the given lockfile. It'll let future PEPs extend the format and installers *must* raise an error when they don't support the major version, but they only *should* warn users when they don't support a minor version and encounter an unknown key.
- `environments`: it's an **optional** key, and it's an array of strings that indicate [environment markers](https://packaging.python.org/en/latest/specifications/dependency-specifiers/#dependency-specifiers-environment-markers) for which the lockfile is compatible with. It's essentially a white‑list that lets installers error early if no environment matches. Obviously, the environments markers *should* be exclusive/non-overlapping. I guess it's not required because some lockfiles are not limited by the environment such is the case of many packages that only depend on pure Python dependencies.
- `requires-python`: it's also an **optional** key, that indicates the minimum viable [version](https://packaging.python.org/en/latest/specifications/version-specifiers/) of Python for which the lockfile is compatible with.
- `extras`: it's an **optional** key to list the different extras (optional dependencies) that the lockfile supports. They can later be referenced in marker expressions.
- `dependency-groups`: similar to `extras`, it's an **optional** key to list the different dependency groups that the lock file supports and they can also be reference in marker expressions down the line.
- `default-groups`: I was not familiar with this feature so if you have a project that utilizes it please let me know 😀 but it's an **optional** key. The groups listed here *must* not be listed in dependency groups. The PEP doesn't say anything about whether they should be mentioned in `extras` as well but I guess it's either superfluous to mention them there or it doesn't make sense. I think this is different from what the [PEP 771 draft](https://peps.python.org/pep-0771/) suggests right? The `default-groups` are used to represent a default install profile and they are *dependency groups* while what PEP 771 suggests to be able to choose some *extras* to come along by default in one wheel or another. So they act at different levels of the packaging (installing from a lock file vs installing a wheel).
- `created-by`: a **required** key that records which tool generated the lockfile. I guess it is required so that, if needed to refresh the lockfile, you know which tool to use. The PEP suggests tools to record the normalized name if it's a Python package. We'd think that adding the tool version would be helpful as well or some other metadata, but I think it's not the goal here in this key the lock file gives the tools the choice of using the `[table]` to record any details they deem necessary to know what inputs were used to create the lock file.



**Note**: The PEP *allows* lockers, tools that write the locks, so they do the resolution, example: uv, to choose to not support writing lock files that support extras and dependency groups, which would make them create single-use lock files only (it's not the case of `uv`. `uv` creates multi-use cross-platform lock files). I guess in that case they'd create many `pylock.<extra_or_dep_group_name>.toml`. If a tool supports extras, it *must* support dependency groups as well and vice-versa. Tools that support extras and dependency groups *should* set the `extras` and `dependency-groups` keys to empty lists in case the project doesn't have one to signal the the lockfile is multi-use though it seems single-use.



Now let's get to the juicy part. The `[[packages]]` array. This is a **required** array of tables and keys, and every element of the array represents a package and some information about it. Packages *might* be repeated but with different data about them and at install time they *must* narrow down to one entry. So you might have different versions of a package for different environment markers or different dependency groups (for example during training machine learning models you might want a very specific version of `torchmetrics` but during evaluation you might want a different version of it, the package torchmetrics would be repeated twice in the `[[packages]]` array).  
  
We'll now get into the nitty gritty details of the format. It's not that interesting so you might jump to the next section. The key idea here is that there are different source blocks that you can use to record details about your package. `vcs` for reproducible source trees that can change between commits. We pin the commit‑id so it’s immutable. `directory` for local folders, typically editable installs. `archive` which is a catch‑all for archives that are not handled by the other source blocks (archives can be anything, wheels, sdists, archives containing source trees etc.), obviously the required thing here is the hash, having other things like the size of the archive would be nice as well if the locker can provide them but it's not required. And then you have the obvious `sdist` and `wheels` as sources of your package. These are clear by themselves, record the different wheels with different markers for a package, and the sdist as a fallback case.  
  
The nitty gritty details that you might skip (or maybe you can read what's the installation routine for each package source):



- Package identification information:
  - A **required** `name` key that is the normalized name of the package.
  - An **optional** `version` key. It is optional and *should* only be included when the version can't change like when the case of an sdist or a wheel. In that case writing the version in the lock file adds useful redundancy (the version is usually included in the artifact like the wheel's name or `METADATA` or something) for humans, but most importantly it is guaranteed to be correct, thus it makes sense to include it. But if that is not guaranteed, then the version must not be included. You have to avoid cases where the version might be inconsistent between the install time and lock time. This is the case of a local directory, a VCS snapshot chosen by commit-hash, a tarball whose build backend derives the version at build time etc. These sources can yield different metadata at every installation. ***Remember***, it's not because the bytes of the source are frozen (by the path, commit-id or URL) that the project's build process will always label the version similarly.
- Markers for when the package can be installed:
  - An **optional** `marker` key to record environment markers for which the package can be installed. Environment markers in this context are the [existing specification](https://packaging.python.org/en/latest/specifications/dependency-specifiers/#dependency-specifiers-environment-markers) + an extension that the PEP defines for this context of lock files only to include extras and dependency groups.
  - An **optional** `requires-python` key that contains the [version specifier](https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers) for this distribution of the package.
- Inclusion of dependency metadata purely for **informational** reasons:
  - The **optional** array of table `[[packages.dependencies]]`. Each table should list enough key/value pairs (`name`, optionally `version` or any source‑defining subtree) to unambiguously refer to another `[[packages]]` entry that is a direct dependency of the current entry. It is purely informational and for auditing purposes. Installers *must not* rely on it for resolution.
- Source definition: exactly one of the five sections below *must* be present (so all of the tables below are optional but the lock file must contain at least one of them). The `vcs`, `directory` and `archive` are mutually exclusive and they're mutually exclusive with `sdist`/`wheels`. These are used to record the source of the package. Notice that they're all tables except for the wheels source which is an array (so that we can record different wheels of the same package): `[packages.vcs]`, `[packages.directory]`, `[packages.archive]`, `[packages.sdist]` but `[[packages.wheels]].` Before we dive into each individual source block, let's talk about the `index` key that is at the same level of the source blocks (so it belongs inside every `[[packages]]` table).
  - packages`.index`: an **optional** key. Base URL of the package index from which the recorded sdist or wheels were fetched (e.g <https://pypi.org/simple/> or a private index etc.). Lockers *should* fill this whenever they know the index URL. It helps tooling (SBOM generators, provenance scanners, cache warmers etc.) and it gives installers one more place to look at if the recorded `url` isn't working or is omitted. Installers *may* use this URL to download the *same* file (validated by hash) and they *must not* switch to a *different* file under any circumstance.
  - `[packages.vcs]`: this an **optional** table. It is used to capture an *immutable checkout* of a version-control repository. Tools *may* decide to not lock or install VCS sources and if they support them they also *may* restrict the accepted VCS types. And both lockers and installers *should* offer users the choice whether to include VCS or not. If this table is set for a package: then the installer will clone the repository to the recorded commit ID, then build inside `subdirectory` (if present) and install it. Installation from it is considered similar to a direct URL reference. This table includes:
    - `type`: a **required** key. It must be one of the VCS registered [here](https://packaging.python.org/en/latest/specifications/direct-url-data-structure/#direct-url-data-structure-registered-vcs).
    - `url`: **required IF** `path` is omitted. It is just the remote repository's location.
    - `path`: **required IF** `url` is omitted. Again, it's just the **relative** path to the repository. The PEP says that *if* a relative path is used, then it *must* be relative to to the lock file. And that relative paths *may* use POSIX separators.
    - `request-revision`: an **optional** key. It's the branch/tag/ref/commit/revision/etc. that the user requested. It *must not* influence checkout. This is purely for auditing purposes.
    - `commit-id`: you guessed it right, it's a **required** key! It identifies the *exact* immutable revision. For hash-supporting VCS backends, a full commit hash *must* be used.
    - `subdirectory`: an **optional** key. It *must* be a relative path from repo root to the actual Python project.
  - `[packages.directory]`: similar to the `vcs` source, this is an **optional** table. It can be used to record a local filesystem tree. It can be used for example for editable installs. Again, lockers and installers `may` refuse to support directory sources and they *should* provide a way for the user to opt-in or opt-out. Installation in this case consists in just building the package and installing it or linking the editable, respecting `subdirectory` if it is present. Installating from a package directory is considered similar to a direct URL reference. This table consists of:
    - `path`: you guessed it right once again, this key is **required**! So either an absolute path to where the package is or a lockfile-relative path. Relative paths *must* be relative to the lock file and POSIX style is *allowed*.
    - `editable`: a boolean **optional** key. It just says whether the package was installed as an editable installed at lock time or not. An install *may* ignore this key if editable mode in a given context is useless (like the case for a container image build).
    - `subdirectory`: Again an **optional** key and it means the same as in `vcs`.
  - `[packages.archive]`: this is an **optional** table to reference a single archive file. It can be a wheel, an sdist or an arbitrary archive (tar/zip) containing a source tree obtained via direct URL *or* local path. Tools *may* refuse to lock or install archive entries and users *should* be able to opt-in or opt-out. During installation of a package referenced by an archive source, the installer fetches the file, validates it, meaning it verifies the file size and all listed hashes, then it builds and installs it. This is also considered a direct URL reference. This table consists of:
    - `url`: **required IF** `path` is omitted. URL of the archive.
    - `path`: **required IF** `url` is omitted. Similar logic to `packages.vcs.path`.
    - `size`: **optional**. It's the size of the archive as integer. Tools *should* include it when it's easily available.
    - `upload-time`: **optional** datetime key that records when the file was uploaded. Both the date and the time *must* be in UTC.
    - `hashes`: a table instead of a simple key! It is **required** and the table *must* contain at least one entry of different hash values (for different algorithms). The hash algorithms *should* be recorded as lowercase and the PEP *recommends* one of the secure algorithms from `hashlib.algorithms_guaranteed` to be always included (specifically at the time of writing, sha256).
    - `subdirectory`: **optional** again. Same role as before.
  - `[packages.sdist]`: **optional** table to describe the source distribution file from an index or a direct URL. Again, tools *may* refuse sdists and they *should* allow users to opt-in / opt-out. The installation process is similar to the rest, download, verify, build, install. The installer *may* fallback to `packages.index` if the `url` isn't working (and I guess it *must not* choose a different file). The table consists of:
    - `name`: **optional IF** the trailing component of `url` or `path` already matches the name of the sdist.
    - `upload-time`: **optional** and similar to `packages.archive.upload-time`.
    - `url`: **required IF** `path` is omitted. Similar to `packages.archive.url`.
    - `path`: **required IF** `url` is omitted. Similar to `packages.archive.path`.
    - `size`: **optional** and similar to `packages.archive.size`.
    - `hashes`: **required** and it is similar to `packages.archive.hashes`.
  - `[[packages.wheels]]`: it is, as we said before, an array of tables. So this is an array that lists every wheel the locker believes relevant for a given package (different platforms, or ABIs or Python versions). Though it is **optional** (because of the mutual exclusive thing with the rest of the sources, remember), tools, both installers and lockers, *must* support wheels, as opposed to the rest of the sources. The installer picks the first wheel whose tags match the target architecture, interpreter, Python version, platform etc. If none match continue to `sdist` logic or *fail* (remember, it can't continue to the rest of the sources because they're mutually exclusive with `wheels`, the only two sources you can combine are `wheels` and `sdists`). Then retrieve the file via `path` (preferred) or `url`, then validate size (if provided) and hashes and then install the wheel. Optionally, the installer ***may*** fall back to `packages.index` if the `url` isn't working or to a tool-specific cache/mirror or other mechanism, butit **must not** choose a different wheel. Which wheel gets installed has to be decided *offline* at lock time for reproducibility. **Each entry** of the array consists of:
    - `name`: similar to sdists.
    - `upload-time`: similar to sdists.
    - `url`: similar to sdists.
    - `path`: similar to sdists.
    - `size`: similar to sdists.
    - `hashes`: similar to sdists.
- Finally, every `[[packages]]` entry *may* (so optional) carry a `[[packages.attestation-identities]]` **array**. On top of the hash and the size information that you can verify about a wheel, you'd might want to verify who built the wheel and how (e.g., the GitHub Actions etc.). This is the role of the attestation. So this is added for security, it's not a signature or a hash but it tells auditers *which* attestation to look for on the index and *who* is expected to have signed it. An index like PyPI may attach to every file it serves an attestation (the attestation though is attached to the package entry). We don't record it in its fullest because that'd be huge, so what this table records instead is the signing identity, so the *who*, for every file it might install. And every table is just a free form of key/value pairs plus a **required** `kind` key, which is a unique label for the signer, e.g., "GitHub". Tools *should* include every attestation identities they find. So installers or auditors can compare what’s in the lock file with what the index returns to ensure files are signed by the expected party. Nothing in the PEP disallows recording different identities for the same package.



**Note**: do not confuse the array of wheels with the array of packages. A package can have different entries in the `packages` array and each entry can have different entries in the `wheels` array. For example, you might want different versions of a package depending on the extra (e.g., `train` extra vs `evaluation` extra for a machine learning project). Since you have two extras with two different versions of the same package, you'll have two listings of that package in the `packages` array.



## How it solves the mentioned issues



One great thing about the standard is how many good practices and things are included by default. The lock file will list the exact resolution that led to your environment along with every artefact that may be installed. It'll also include hashes by default (and tools should use size as well). You can "easily" (if the installer supports it) include a default index fallback in case the `url` for fetching the sdist or the wheel of a package isn't working and the installer will still use the filename, the version and the hashes to verify that everything is good.  
  
It is easy to have one lock file that contains everything about your different use cases: inclusion of extras and dependency groups in the same lock file, environment markers etc. You do have the choice (maybe through the tool you use) to have different lock file for different use cases, but at leat you know they're all standardized and follow the same format and can be used by any tool downstream.  
  
Lock files, though much complex than a plain `requirements.txt` text file, they're much more auditable. This is not thanks to the TOML file which only provides more readability compared to other formats, but thanks to all the metadata and information that is included by default in the lock file itself, instead of having to rely on comments or whatever to understand why a package is included in the `requirments.txt`.  
  
Finally, though it is still at its beginning, the lock file supports attestation identities. Not to say this is complete or perfect, but it's an upgrade compared to what existed before and it should be appreciated.



## pylock.toml is not perfect yet



I don't see a lot of issues with the current standard so you're very welcome to tell me about your pain points with the current lock file experience. It is still relatively new so I doubt many people have tried it but if you have any complementary information do not hesitate to tell me about it.  
  
My biggest issue with the current `pylock.toml`, and I totally understand why it is an issue, I'm not really complaining about it, it is its dependence on the resolver. The lock file is agnostic with respect to the installer. Any installer that chooses to support PEP 751 can take the lock file and use it, but the lock file inherently depends on the resolver. So two different resolvers may (or may not) lead to two different lock files. And in a similar manner, nothing makes the lock file cross-platform. We know that some tool lock files such as `uv`'s or `Poetry`'s are cross-platform (though I think their strategy of achieving that is different), but nothing imposes that on a `pylock.toml` (or any `pylock.*.toml` file) generated in accordance with PEP 751. So if we go back to one of the issues we mentioned before of doing `pip freeze` on a platform and not ensuring reproducibility of the environment for missing a dependency that might be required on a different platform, we still do not necessarily solve it *by default*. It all depends on the resolver.  
  
In a similar fashion, though a minor issue, there is no standard way to reference the groups in `default-groups`. Every locker may include its own synthetic group that stands for what the default group stand for. That makes installer experience inconsistent.  
  
Besides that, from what I have gathered with my experiences and reading (especially the uv repo issues), the main limitation is the workspace support that is missing from PEP 751's lockfile. We'll see that more in the next section. I'm including this in the next section because I really like [`uv` workspace model](https://docs.astral.sh/uv/concepts/projects/workspaces/#getting-started).  
  
And besides these two things, there are just some minor things here and there. Like the PEP itself mentions that the current lock file isn't a drop-in replacement of `requirements.txt`, see [Semantic differences with `requirements.txt` files](https://peps.python.org/pep-0751/#semantic-differences-with-requirements-txt-files). It's because `requirements.txt` support specifying installation options at install-time, can reference other requirements files via `-r`, can interpolate environment variables.  
  
In the same vein of things that would be great to have more control over, it's the editable installs. For the moment its inclusion through the `editable` key may be ignored by the installer. Maybe having more control on when or when not to ingore it would be nice to have instead of letting the installer infer when it is useful or unnecessary.  
  
And finally, and I think this is a hard issue to solve, cross-platform builds from sdists can still differ from one build to another depending on compiler flags etc.



## uv.lock vs pylock.toml



### workspaces



As we said before, `pylock.toml` lacks the support for workspaces. Let's see that. For that I'll take the structure of the example in [uv's documentation about workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/#getting-started). Let's do it this way (at the time of writing this, I'm using uv version v0.6.17)



```
uv init albatross
cd albatross
uv init bird-feeder  // will initialize bird-feeder as workspace member of albatross
uv init seeds  // will initialize seeds as workspace member of albatross
```



You should then have a `pyproject.toml` that looks like the following and that indicates that `bird-feeder` and `seeds` are actual members of your workspace.



```
[project]
name = "albatross"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = []

[tool.uv.workspace]
members = [
    "bird-feeder",
    "seeds",
]
```



Now, let's do this:



```
uv add bird-feeder
```



This will add `bird-feeder` as a dependency to `albatross` and, quoting the `uv`'s documentation: "The `workspace = true` key-value pair in the `tool.uv.sources` table indicates the bird-feeder dependency should be provided by the workspace, rather than fetched from PyPI or another registry". Your `pyproject.toml` should look like this:



```
[project]
name = "albatross"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "bird-feeder",
]

[tool.uv.workspace]
members = [
    "bird-feeder",
    "seeds",
]

[tool.uv.sources]
bird-feeder = { workspace = true }
```



Let's stop here for the moment and let's look at the generated lock file, `uv.lock`:



```
version = 1
revision = 2
requires-python = ">=3.13"

[manifest]
members = [
    "albatross",
    "bird-feeder",
    "seeds",
]

[[package]]
name = "albatross"
version = "0.1.0"
source = { virtual = "." }
dependencies = [
    { name = "bird-feeder" },
]

[package.metadata]
requires-dist = [{ name = "bird-feeder", virtual = "bird-feeder" }]

[[package]]
name = "bird-feeder"
version = "0.1.0"
source = { virtual = "bird-feeder" }

[[package]]
name = "seeds"
version = "0.1.0"
source = { virtual = "seeds" }
```



Let's export that `uv.lock` to a `pylock.toml` through the `uv export --format pylock.toml`. You'll get this:



```
# This file was autogenerated by uv via the following command:
#    uv export --format pylock.toml
lock-version = "1.0"
created-by = "uv"
requires-python = ">=3.13"
```



### Partial dependency graph installs



The other main difference between both is the ability to select different entry points in the dependency graph. `uv` allows that through its `uv.lock` (and it being the installer on top of it). What do we mean by that? Let's take an example. Let's generate a project with a dependency group:



```
uv init exp_proj
cd exp_proj
uv add pillow  # or any other cached package you have
uv add --group dataframes pandas
```



This should get you a `pyproject.toml` that looks like:



```
[project]
name = "exp-proj"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "pillow>=11.2.1",
]

[dependency-groups]
dataframes = [
    "pandas>=2.2.3",
]
```



If we look at the `uv.lock` for this:



```
version = 1
revision = 2
requires-python = ">=3.13"

[[package]]
name = "exp-proj"
version = "0.1.0"
source = { virtual = "." }
dependencies = [
    { name = "pillow" },
]

[package.dev-dependencies]
dataframes = [
    { name = "pandas" },
]

[package.metadata]
requires-dist = [{ name = "pillow", specifier = ">=11.2.1" }]

[package.metadata.requires-dev]
dataframes = [{ name = "pandas", specifier = ">=2.2.3" }]

[[package]]
name = "numpy"
version = "2.2.5"
source = { registry = "https://pypi.org/simple" }
sdist = { url = "https://files.pythonhosted.org/packages/dc/b2/ce4b867d8cd9c0ee84938ae1e6a6f7926ebf928c9090d036fc3c6a04f946/numpy-2.2.5.tar.gz", hash = "sha256:a9c0d994680cd991b1cb772e8b297340085466a6fe964bc9d4e80f5e2f43c291", size = 20273920, upload_time = "2025-04-19T23:27:42.561Z" }
wheels = [
    { url = "https://files.pythonhosted.org/packages/e2/a0/0aa7f0f4509a2e07bd7a509042967c2fab635690d4f48c6c7b3afd4f448c/numpy-2.2.5-cp313-cp313-macosx_10_13_x86_64.whl", hash = "sha256:059b51b658f4414fff78c6d7b1b4e18283ab5fa56d270ff212d5ba0c561846f4", size = 20935102, upload_time = "2025-04-19T22:41:16.234Z" },
    { url = "https://files.pythonhosted.org/packages/7e/e4/a6a9f4537542912ec513185396fce52cdd45bdcf3e9d921ab02a93ca5aa9/numpy-2.2.5-cp313-cp313-macosx_11_0_arm64.whl", hash = "sha256:47f9ed103af0bc63182609044b0490747e03bd20a67e391192dde119bf43d52f", size = 14191709, upload_time = "2025-04-19T22:41:38.472Z" },
    ...
]

[[package]]
name = "pandas"
version = "2.2.3"
source = { registry = "https://pypi.org/simple" }
dependencies = [
    { name = "numpy" },
    { name = "python-dateutil" },
    { name = "pytz" },
    { name = "tzdata" },
]
sdist = { url = "https://files.pythonhosted.org/packages/9c/d6/9f8431bacc2e19dca897724cd097b1bb224a6ad5433784a44b587c7c13af/pandas-2.2.3.tar.gz", hash = "sha256:4f18ba62b61d7e192368b84517265a99b4d7ee8912f8708660fb4a366cc82667", size = 4399213, upload_time = "2024-09-20T13:10:04.827Z" }
wheels = [
    { url = "https://files.pythonhosted.org/packages/64/22/3b8f4e0ed70644e85cfdcd57454686b9057c6c38d2f74fe4b8bc2527214a/pandas-2.2.3-cp313-cp313-macosx_10_13_x86_64.whl", hash = "sha256:f00d1345d84d8c86a63e476bb4955e46458b304b9575dcf71102b5c705320015", size = 12477643, upload_time = "2024-09-20T13:09:25.522Z" },
    { url = "https://files.pythonhosted.org/packages/e4/93/b3f5d1838500e22c8d793625da672f3eec046b1a99257666c94446969282/pandas-2.2.3-cp313-cp313-macosx_11_0_arm64.whl", hash = "sha256:3508d914817e153ad359d7e069d752cdd736a247c322d932eb89e6bc84217f28", size = 11281573, upload_time = "2024-09-20T13:09:28.012Z" },
    ...
]

[[package]]
name = "pillow"
version = "11.2.1"
source = { registry = "https://pypi.org/simple" }
sdist = { url = "https://files.pythonhosted.org/packages/af/cb/bb5c01fcd2a69335b86c22142b2bccfc3464087efb7fd382eee5ffc7fdf7/pillow-11.2.1.tar.gz", hash = "sha256:a64dd61998416367b7ef979b73d3a85853ba9bec4c2925f74e588879a58716b6", size = 47026707, upload_time = "2025-04-12T17:50:03.289Z" }
wheels = [
    { url = "https://files.pythonhosted.org/packages/36/9c/447528ee3776e7ab8897fe33697a7ff3f0475bb490c5ac1456a03dc57956/pillow-11.2.1-cp313-cp313-macosx_10_13_x86_64.whl", hash = "sha256:fdec757fea0b793056419bca3e9932eb2b0ceec90ef4813ea4c1e072c389eb28", size = 3190098, upload_time = "2025-04-12T17:48:23.915Z" },
    { url = "https://files.pythonhosted.org/packages/b5/09/29d5cd052f7566a63e5b506fac9c60526e9ecc553825551333e1e18a4858/pillow-11.2.1-cp313-cp313-macosx_11_0_arm64.whl", hash = "sha256:b0e130705d568e2f43a17bcbe74d90958e8a16263868a12c3e0d9c8162690830", size = 3030166, upload_time = "2025-04-12T17:48:25.738Z" },
    ...
]

[[package]]
name = "python-dateutil"
version = "2.9.0.post0"
source = { registry = "https://pypi.org/simple" }
dependencies = [
    { name = "six" },
]
sdist = { url = "https://files.pythonhosted.org/packages/66/c0/0c8b6ad9f17a802ee498c46e004a0eb49bc148f2fd230864601a86dcf6db/python-dateutil-2.9.0.post0.tar.gz", hash = "sha256:37dd54208da7e1cd875388217d5e00ebd4179249f90fb72437e91a35459a0ad3", size = 342432, upload_time = "2024-03-01T18:36:20.211Z" }
wheels = [
    { url = "https://files.pythonhosted.org/packages/ec/57/56b9bcc3c9c6a792fcbaf139543cee77261f3651ca9da0c93f5c1221264b/python_dateutil-2.9.0.post0-py2.py3-none-any.whl", hash = "sha256:a8b2bc7bffae282281c8140a97d3aa9c14da0b136dfe83f850eea9a5f7470427", size = 229892, upload_time = "2024-03-01T18:36:18.57Z" },
]

[[package]]
name = "pytz"
version = "2025.2"
source = { registry = "https://pypi.org/simple" }
sdist = { url = "https://files.pythonhosted.org/packages/f8/bf/abbd3cdfb8fbc7fb3d4d38d320f2441b1e7cbe29be4f23797b4a2b5d8aac/pytz-2025.2.tar.gz", hash = "sha256:360b9e3dbb49a209c21ad61809c7fb453643e048b38924c765813546746e81c3", size = 320884, upload_time = "2025-03-25T02:25:00.538Z" }
wheels = [
    { url = "https://files.pythonhosted.org/packages/81/c4/34e93fe5f5429d7570ec1fa436f1986fb1f00c3e0f43a589fe2bbcd22c3f/pytz-2025.2-py2.py3-none-any.whl", hash = "sha256:5ddf76296dd8c44c26eb8f4b6f35488f3ccbf6fbbd7adee0b7262d43f0ec2f00", size = 509225, upload_time = "2025-03-25T02:24:58.468Z" },
]

[[package]]
name = "six"
version = "1.17.0"
source = { registry = "https://pypi.org/simple" }
sdist = { url = "https://files.pythonhosted.org/packages/94/e7/b2c673351809dca68a0e064b6af791aa332cf192da575fd474ed7d6f16a2/six-1.17.0.tar.gz", hash = "sha256:ff70335d468e7eb6ec65b95b99d3a2836546063f63acc5171de367e834932a81", size = 34031, upload_time = "2024-12-04T17:35:28.174Z" }
wheels = [
    { url = "https://files.pythonhosted.org/packages/b7/ce/149a00dd41f10bc29e5921b496af8b574d8413afcd5e30dfa0ed46c2cc5e/six-1.17.0-py2.py3-none-any.whl", hash = "sha256:4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274", size = 11050, upload_time = "2024-12-04T17:35:26.475Z" },
]

[[package]]
name = "tzdata"
version = "2025.2"
source = { registry = "https://pypi.org/simple" }
sdist = { url = "https://files.pythonhosted.org/packages/95/32/1a225d6164441be760d75c2c42e2780dc0873fe382da3e98a2e1e48361e5/tzdata-2025.2.tar.gz", hash = "sha256:b60a638fcc0daffadf82fe0f57e53d06bdec2f36c4df66280ae79bce6bd6f2b9", size = 196380, upload_time = "2025-03-23T13:54:43.652Z" }
wheels = [
    { url = "https://files.pythonhosted.org/packages/5c/23/c7abc0ca0a1526a0774eca151daeb8de62ec457e77262b66b359c3c7679e/tzdata-2025.2-py2.py3-none-any.whl", hash = "sha256:1a403fada01ff9221ca8044d701868fa132215d84beb92242d9acd2147f667a8", size = 347839, upload_time = "2025-03-23T13:54:41.845Z" },
]
```



You see, every package **points** to all of its dependencies (if there are any, e.g., `tzdata` doesn't require anything else to be installed). So it means that, in a fresh install, you can set up an arbitrary point in the graph and be able to install only that subset of the whole dependency graph. Example, let's delete the venv that was created `rm -rf .venv`. And try this: `uv sync --only-group dataframes` and it'll only install `pandas` as well as its dependencies!  
  
This is not currently possible with `pylock.toml`. I don't think there are many things to add to it in order to be able to do that. We already have a `dependencies` key. It is though optional and the PEP clearly says that its goal, at least for the moment, is purely informational and must not be used by tools. As Charlie Marsh says in <https://github.com/astral-sh/uv/issues/12584>, the decision to not support that in PEP 751 is just to make this first iteration less complex.



### Other (minor) differences



For the other differences that I could spot, is that, for the moment, `pylock.toml` supports attestation identities but `uv.lock` doesn't.



The rest of the differences are mainly due to how the files are written (like how to record the packages entries etc.). And, obviously, PEP 751 allows for using different lock files and for file discovery while uv uses uv.lock.



There are certainly other things that I have missed. If there are ny things that you want me to add to this blog post or you think should be mentioned, please feel free to comment here or comment on Reddit / X. Thank you!



One final word about `uv`, currently it only supports exporting to single use lock files. So if you do `uv export --format pylock.toml` in the project above, you won't see the packages from the `dataframes` dependency group. You should do `uv export --format pylock.toml --group dataframes`. And you'll notice that the generated `pylock.toml` is single-use as well (no `dependency-groups` key or markers), so you might want to save it as `pylock.dataframes.toml`.   
  
Not to say this is bad, I only wanted to mention it in case people get confused with their generated `pylock.toml` files and hopefully to reduce the number of people asking that question on their Discord / GitHub repo.



# Some of the hard problems of Python packaging



In the following sections I wanted to present some of the issues that I find hard when it comes to Python packaging.   
  
One excellent resource that delves into much more specific details about these issues and that I learned about recently is: <https://pypackaging-native.github.io/>.



## An naive introduction to native dependencies from a packaging perspective



To understand the issues that arise in Python packaging when it comes to native dependencies, we have to first understand some concepts. Those concepts mainly pertain to the native dependencies so this part might be a little bit abstract or incomprehensible for you if you have never touched that part. But I'll do my best here and you tell me if there is something you don't understand or if you had a hard time following along.  
  
So what do we call by native dependencies? There is no general rule I think, it's every non-Python dependency, in general a compiled one. Some of the famous libraries that have native dependencies are NumPy, SciPy, torch etc. Not all scientific computing libraries have native dependencies, but almost all the fundamental ones do. What do we mean by that? It means that NumPy has code in a non-Python language and Python only acts as a glue or as a wrapper over to allow us easier usage of the functionality that NumPy wants us to have access to. That's what we call in general Python extension modules. If you go to the NumPy [repo](https://github.com/numpy/numpy), you'll find a lot of code in C. Example for `multiarray`: <https://github.com/numpy/numpy/tree/main/numpy/_core/src/multiarray>.  
  
To understand the problem with native dependencies, let's say you want to develop an extension module for Python using C. We'll then see if you want to do some high performance computing stuff (I only use scientific computing as examples because that's what I'm familiar with but I guess there are use cases in other fields). So you write your C code in a certain manner, for example to expose your module you have to write this `PyInit_{your_module_name}` C function etc. Anyways, there's a way to do things, you can read more about it here <https://docs.python.org/3/extending/extending.html>. It's not important for understanding the rest. Ultimately, your code depends on some things to run. The simplest things your code might depend on is the C standard library. Say you use `glibc` in your code. Now, since your code depends on this, you'll need a way to have the consumers of your package also have this dependency. For the moment, let's stick with `glibc` only as a dependency and we'll introduce more complexity to this later on.  
  
There are two ways to deal with this, and they are just the two ways of how C programs (and I guess other low level languages as well but I'm not familiar with them) work. You can either statically link the dependencies in your code, or have them dynamically linked at runtime. What do we mean by that? well statically linking just means copying the code of your dependencies in your own code (this is a big simplification of how things work but ultimately leads down to this for the end product). It's the linker who does that, not you manually copy-pasting all the code you depend on. This means that should you use only `printf` in your Python extension module, you'll have to copy everything that's related to it in your code. This leads to huge binaries (well not in the case of just using `printf`...). Not good, but at least you are sure that your code is self-sufficient since it has everything it needs contained within it.   
The other way of dealing with this, is to use shared libraries. They have the `.so` extension (shared object). These libraries are linked at runtime, not at compile time like static libraries (also called archive libraries, `.a`). So you don't have to copy their code. This is the preferred way in general and your binaries are lighter, but we go back to having to make sure that the consumers of your code do have these libraries. And in this case, there are different situations with different levels of complexity.  
  
The first situation is, you only care about environments where you are sure that the dependencies of your code exist. Example, `manylinux`. If you only care about the platforms that are included in it, then you'll more or less have an easier time because you know that all those platforms will have a `libc.so.6`, a `libgcc_s.so.1` etc. So you build your wheel within the docker for manylinux and you're good to go. Your final wheel will have `.so` files that represent your Python extension modules but will not contain anything that those extension modules depend on, because that will be linked at runtime.  
  
Now, what if you want to build your wheel for a platform that's not "standardised" or for which you are not sure your dependencies will be present or stay the same across different "versions" of that platform. Or, similarly, what if you depend on a library that has different implementations across different environments, example, [BLAS](https://www.netlib.org/blas/). A very short, incomplete introduction to BLAS is, BLAS is a set of linear algebra operations. It's a specification, and has a reference implementation in Fortran. You can find it on [Netlib](https://www.netlib.org/blas/). Now, BLAS is old and it currently has different other implementations, some that are open source like [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) and some are proprietary like [Intel's MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html). Now, since you can't ensure this is present in your end users's environment, the only way to handle this is to ship the shared library within your wheel. This might cause some headaches that we'll talk about later. But first, let's see a concrete case of this, with NumPy. I'll just `uv add numpy` in a Python project that I initialized with `uv init`. You may just do the classic `python3 -m venv .venv` into `source .venv/bin/activate` into `pip install numpy`. At the time of writing, and on my current laptop, it's the following wheel that gets installed `numpy-2.2.5-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` but the concepts should be similar to you. Anyways. Whether you look into `site-packages` or you extract the wheel itself, you should find a `numpy.libs`. Those are the non-Python dependencies that NumPy ships with its wheel. In mine I can find `libscipy_openblas64_-6bb31eeb.so`. We find other `.so` files in `numpy.libs`. They are all shared libraries that NumPy ships because it's unsure that it'll find the same implementations (or, ABI, we'll talk about that later) across the different platforms in manylinux. And if you look in the `numpy` directory, you'll find other `.so` files. Those are the extension modules. Example: `numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so`. We can find what are the required shared libraries of this extension module by using `ldd`, so doing `ldd .venv/lib/python3.13/site-packages/numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so` you should get something like:



```
linux-vdso.so.1 (0x00007fffb19f2000)
libscipy_openblas64_-6bb31eeb.so => /home/username/test/.venv/lib/python3.13/site-packages/numpy/_core/../../numpy.libs/libscipy_openblas64_-6bb31eeb.so (0x00007c7a79400000)
libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007c7a79000000)
libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007c7a7a917000)
libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007c7a7b408000)
libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007c7a78c00000)
/lib64/ld-linux-x86-64.so.2 (0x00007c7a7b44d000)
libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007c7a7b401000)
libgfortran-040039e1-0352e75f.so.5.0.0 => /home/username/test/.venv/lib/python3.13/site-packages/numpy/_core/../../numpy.libs/libgfortran-040039e1-0352e75f.so.5.0.0 (0x00007c7a78600000)
libquadmath-96973f99-934c22de.so.0.0.0 => /home/username/test/.venv/lib/python3.13/site-packages/numpy/_core/../../numpy.libs/libquadmath-96973f99-934c22de.so.0.0.0 (0x00007c7a78200000)
libz.so.1 => /lib/x86_64-linux-gnu/libz.so.1 (0x00007c7a7b3e5000)
```



So those are all the shared libararies that the `_multiarray_umath.so` extension module depends on. Notice how some are looked for in the `numpy.libs` directory (the libscipy\_openblas, libgfortran and libquadmath shared libraries) while others are on my system. Those on the system are similar across platforms under `manylinux`. The rest are shipped with NumPy.  
  
This causes a lot of issues. And we have issues that concern authors packages and issues that concern consumers of packages. And in both cases we can sub-divide into issues related to sdists only and issues related to wheels.  
  
Let's start with sdists, since that is the source of truth and what is used to build your package into wheels. But that is also a source of problems. There's no way, at the moment, to specify the non-Python build requirements of your project, neither at the sdist level nor at the wheel level. I'm talking external shared libraries, compiler version, compiler flags etc. Whatever non-Python requirement you might think of. So when you expose the sdist to users, they're at risk of either failing the build or potentially mess up the package manager's cache. If the build fails, because they don't have the necessary requirements to build the wheel, in many cases there is no systematic way to inform them *correctly* of what's happening because there is no way to communicate all of your non-Python build requirements to begin with, though some of the issues might be picked up by the build system. And if the build passes and they happen to rebuild a new wheel from the same sdist but with, say a different compiler version, then they'll use the old wheel because wheels do not contain any information about non-Python build requirements again so the cache doesn't know about them (I think for `pip` it only looks at the wheel name but I'm not totally sure about the other package managers, that's why it's a potential risk). So you as a package author are required to upload the sdist on PyPI because it is the source of truth behind your package and redistributors will probably need it to repackage it to their sauce but at the same time, it can lead to many direct users of your package getting into all kind of issues.  
  
When uploading wheels. Well first, let's discuss about how all of us are expecting wheels from every package we use. Just a few days ago I was complaining about a package requiring to build from source and which didn't work even on a very common environment for that package and complaning why they didn't just provide wheels. But providing wheels is not easy. Not easy at all. There's is a huge amount of different configurations (think all CPython minor versions, platform, architecture etc.) for which you have to build for, the process requires both finance resources and human resources to both run and maintain the different CI pipelines to build these wheels. On top of that, sometimes you can't just vendor in some libraries due to licenses and what not, and at the same time you can't express your non-Python build requirements because the actual wheel metadata doesn't allow for that. And let's say everything goes well and you have built a wheel for your users, they're happy, they install it. It's not the end of their problems. Because what if they install in the same virtual environment a different packages that uses a shared library that has the same API as the one you vendored in your wheel but with a different ABI? Meaning that both this other package's shared library `other_foo.so`, and your shared library, `your_foo.so`, expose the same symbols with the same signature etc., but the "physical" execution is different, maybe a different alignment is used because they different compiler flags were used to generate them etc. This can cause a lot of issues.  
  
One of the "potential" solutions to this is what's called a build-farm. You can learn more about it here: <https://pypackaging-native.github.io/meta-topics/no_build_farm/> but quickly it just means an operating public and shared "factory" where sdists come in and get built in well defined reproducible environments for every relevant platform/architecture that the build farm cares about. And like the resulting wheels can get sanity checked, signed before finally being uploaded to PyPI. This would help a lot because now users are sure, at least those on the platforms that the build farm cares about, to get working wheels as soon as an sdist is available, instead of unconsciously fetching an sdist with `pip` and building it into a potentially "bad" wheel. On top of that, a build farm shares the costs among the package authors. No more everyone running large multiple CIs individually for their wheels and the best of all, is that the shared libraries with which wheels are produced are the same. So you're almost sure that the wheels from the build farm will not have issues from that regard when used in the same environment. And obviously we'd get all the benefit from "easily" rebuilding wheels when there is an upgrade in CPython or something, and having some good sanity checks that are applied to every wheel before it being uploaded on PyPI. One other scenario, I think that's what Conda does but let me know if I'm wrong, is to also ship the shared libraries themselves and let wheels link to them at runtime instead of having to vendor them within the wheels. But I think that requires different discussions as well because Conda is a system's package manager while that's not the case for `pip` or `uv` etc. (well, `uv` can manage Python versions so it's not exactly like `pip` but you get the gist). A build farm is not the holy grail though, I'm not trying to say that. I only mention it to get you familiar with some of the issues and how some solutions can solve them. Obviously a build farm requires massive investments, both financial and human capital. There are security and supply-chain risks. There are only a handful (well more in the thousands) of packages that would require all the power of a build farm, the rest of the packages not really, and deciding which falls under which category is not an easy task. There are still limitations in the metadata and a build farm doesn't magically solve that. And similarly to this last issue, it doesn't solve the problems of accelerator-dependent packages, which I'm gonna talk about next.



# A naive introduction to issues with accelerator-dependent packages



I'll start with what I'd like to see someday. Instead of having to write the following `pyproject.toml`:



```
[project]
name = "my-package"
version = "0.1.0"
description = "Add your description here"
requires-python = ">=3.12"
dependencies = []

[project.optional-dependencies]
cpu = [
  "torch>=2.6.0",
]
cu118 = [
  "torch>=2.6.0",
]
cu121 = [
  "torch>=2.6.0",
]
cu124 = [
  "torch>=2.6.0",
]

[tool.uv]
conflicts = [
  [
    { extra = "cpu" },
    { extra = "cu118" },
    { extra = "cu121" },
    { extra = "cu124" },
  ],
]

[tool.uv.sources]
torch = [
  { index = "pytorch-cpu",   extra = "cpu"   },
  { index = "pytorch-cu118", extra = "cu118" },
  { index = "pytorch-cu121", extra = "cu121" },
  { index = "pytorch-cu124", extra = "cu124" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```



I'd write something much more simpler. I'd just have to state that my package is dependent on `torch`. And let the package manager do its magic. Obviously, this is a hard thing to do, like even if we have solved every packaging problem, if an environment has CUDA, how would the package manager should which version of CUDA to install (if there is compatibility between some) and how would it choose between CUDA and CPU. So I think we'd still require to involve the user somehow, but maybe that involvement can be standardised. In my case, I have to document and communicate to users that to get this or that version of pytorch, use this or that `extra`.



This is not the only issue with accelerator-dependent packages though. Again, you can read a lot about it on <https://pypackaging-native.github.io/key-issues/gpus/>. But, the core of the problem is that the thing we need to express in metadata, like "this wheel needs CUDA x.y at runtime" and "this will not work on older drivers" is simply inexpressible today. The same as the native dependencies we talked about above. But this issue is more complicated than that. This needs CUDA x.y is a good abstraction but behind the abstraction lies a lot of issues. CUDA consists of three main components, as detailed by the link above:  
- The kernel-mode driver (KMD): the `nvidia.ko` module that actually talks to the hardware. As long as you are above NVIDIA’s minimum version requirement this piece is rarely the limiting factor for Python packages.  
- The user-mode driver (UMD): it's the `libcuda.so` library. Every CUDA program, Python or not, goes through that library when it finally wants to do work on the card.  
- The CUDA Toolkit (CTK): this is what we download either from NVIDIA or as libraries like `nvidia-cublas-cu11` etc. The CTK starts with the CUDA runtime, `libcudart.so` and can contain other libraries like cudnn for deep learning kernels, cublas for linear algebra, cufft for fast Fourier transforms etc. (that's how you get the torch wheel torch-2.7.0+cu128-cp313-cp313-manylinux\_2\_28\_x86\_64.whl weighing 1.1GB). And some developer tools like `nvcc` and `nvrtc` which is a JIT compiler and `nvJitLink`.  
  
And obviously none of those components can be expressed currently in the wheel metadata. And since this is a very complex issue, one way to reduce the complexity is to look at what compatibility guarantees do we have.  
- The first one and easiest one is the driver forward compatibility also called binary compatibility. A program that works with driver X will also work with any newer driver Y > X.  
- The minor version compatibility (MVC) introduced with CUDA 11. So this is compatibility across minor versions for the same major family. Any code built against any CUDA 11.x runtime is promised to run on any CUDA 11.y driver. This helps a little bit because it's easier to update CTK than the driver.  
- With CUDA 11 as well, binaries built with a newer CTK will run on any older CTK from the same major series as long as the program does not rely on features introduced after that older minor version. So code built with 11.8 will run on 11.2 if it doesn't use features that were introduced after 11.2.  
  
All of these guarantees together constitute the CUDA Enhanced Compatibility, CEC.  
  
There is a catch though. The guarantees above apply only to code that is fully compiled down to SASS, Nvidia's machine instruction set. They do *not* apply to PTX, the higher-level intermediate representation that the driver can JIT-compile at run time, nor to code compiled on the fly with `nvrtc`. PTX produced with CTK 11.8, for example, may fail on a system that has only a CUDA 11.2 driver.  
  
This is somewhat solved with CUDA 12 brings through the `nvJitLink` library which can JIT-link PTX into SASS in an MVC compatible way. But, that means that's one more dependency to add right. So Python wheels that want to benefit must start using `nvJitLink` (the `pynvjitlink` wrapper is one option).  
  
This section was very CUDA-heavy because I'm not familiar with the rest of the ecosystem (ROCm/AMD etc.). I guess they have similar problems, but don't take my word for it. I hope I gave you a good overview of how much headache it is to deal with accelerator-dependent packages. Some potential solutions would be to improve the metadata in wheels and improve the environment markers and selectors for packages. And also to do better cross-registry plumbing so that even if you use `pip` it's easier to know where to get the CUDA pieces or something.



## Conclusion



That's all I wanted to talk about since as a machine learning engineer I mostly encounter or I am mostly familiar with the native dependencies issues and with the packages dependent on accelerators. But there are many other different problems that you can read more about here <https://pypackaging-native.github.io/>. I didn't want to stop on problems, so I decided to include a small section where I point to spaces where solutions are discussed.



# Where to find some of the initiatives



Obviously, I do not know every initiative and ongoing effort to solve this issue and I do not claim that I do know that, but the goal of this section is just to get you to discover some of the spaces where you can read more and discover what the community is trying to do about this, or at least, those that can make an effective change.  
  
I think most of you know about DPO, [Discuss Python Org](https://discuss.python.org/). But I'm mentioning it here just in case, it's a "General help/discussion forum for the *Python* programming language". And within it, you can find the [Packaging](https://discuss.python.org/c/packaging/14) sub-forum. For every PEP that exists, you'll find at least one very length thread about it. Many PEPs span multiple threads. You'll also get to know other people issues through the reasons why a PEP is suggested etc. To be honest I could never get past the initial few discussions in a "historic" discussion, the PEPs are almost always self-sufficient and I find it hard to read threads of conversation. But it's just me, I'm sure a lot would enjoy doing archeology in those threads 😁 do not hesitate to tell me what you learn!​  
  
One of the places that I didn't know about, thanks to [Charlie Marsh](https://github.com/charliermarsh) for telling me on `uv`'s Discord server is: <https://wheelnext.dev/> (I learned about <https://pypackaging-native.github.io/> through it).  
  
That's it. I'll update this section as I discover more things. I took the deliberate choice of not explaining the PEP drafts that you'll find say on [wheelnext](https://wheelnext.dev/) or on <https://peps.python.org/topic/packaging/> because there are, in general, a lot of discussions and debate around an open PEP an I doubt I can present an open PEP without presenting every side, argument, statement etc., and that's a phenomenal work and it's better represented by the threads around it.  
  
It feels a bit off to end on a section this small, so I'll just add a bonus section.



# Bonus: Understanding dependency specifiers (and environment markers) + pylock.toml extension



Specifications: [Dependency specifiers](https://packaging.python.org/en/latest/specifications/dependency-specifiers/), [Version specifiers](https://packaging.python.org/en/latest/specifications/version-specifiers/)



I'm just putting this here because I noticed when talking to people about Python packaging and especially when it comes to the more practical day-to-day stuff, they don't understand some basic things like dependencies specifiers or wheel names etc. I believe I have talked about wheel names in the first part.



I won't get into the nitty-gritty details of the grammar for dependency specifiers, you can read about it in the specification's page. My goal here is just to explain what are dependency specifiers, what's their goal, how they are used, and understand the environment markers part and explain the extension brought by PEP 751.



So first, let's start with version specifiers. As the name suggests, it's a way to specify the version of something, in our case it'll be Python packages. I always recommend reading the PEPs and specifications yourselves, they're, in most cases, easy and nice to read. In this case you'd gain some insight in how to version your package if someday you release one 😀 but to make it short, the version specifier's goal is to know **which version of a package you want**. You can specify that using comparison operators, example : `some-package>=1.0,!=1.1.*`. Both `>=1.0` and `!=1.1.*` are called *version clauses*. The comma indicates the logical operator `AND`. So both clauses describe a set of allowed versions. It's nothing more than that, it only describes which versions of the given package are allowed to be installed. It says nothing about which wheel for which platform to fetch, whether or not to fetch the extras of the package or not (example whether to install the `excel` extra of `pandas` or not), from which index to fetch the wheel etc. That's **the role of dependency specifiers**.



So you can see dependency specifiers as an extension of version specifiers to bring in extra information, namely the source of the package, the extras, and the environments conditions which govern when the package should be installed. These environment conditions are specified through **environment markers** which say for example which system platform you are on (Windows, Linux etc.), which Python implementation, which Python version etc. That's all for dependency specifiers, they're really just a complete requirement line that embeds a version specifier and augments it with extras, potentially direct references and marker expressions.  
  
So there are two ways to specify a dependency, either with a name or with a direct reference. Not both.  
  
The example from the specification for a name based lookup is: `requests [security,tests] >= 2.8.1, == 2.8.* ; python_version < "2.7"`. You can read this as as telling the installer (`pip` or `uv` or whatever), if the current environment of execution has a Python version < 2.7, then the allowed `requests` versions you can install are those later than 2.8.1 and matching the 2.8 release segment. So version 2.8.0 is not allowed, versions later than 2.9 are not allowed as well. On top of that, you tell the installer to install the `security` and `tests` extras as well! If you don't already have a resolution (from a lock file), then you'll have to wait for the installer to make its dependency resolution before knowing which exact version of `requests` will be installed. For the moment you just know that it'll fall in that set computed from your version specifiers. The environment markers act as early filters or as guidance for which packages and from which sources to fetch under which circumstances (maybe you require an extra package only under Windows or maybe for backwards compatibility you require some old packages that are now part of the standard library etc.).  
  
If you want to specify a dependency using a URL, let's say `pandas`, then you can do something like `pandas[excel] @ https://files.pythonhosted.org/packages/e8/31/aa8da88ca0eadbabd0a639788a6da13bb2ff6edbbb9f29aa786450a30a91/pandas-2.2.3-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`  
This points to the wheel of pandas version 2.2.3 that was built for Linux distributions, x86\_64 arch and for CPython 3.13. And we ask the installer to install the `excel` extra as well. We can add another marker, for illustrative purposes: `pandas[excel] @ https://files.pythonhosted.org/packages/e8/31/aa8da88ca0eadbabd0a639788a6da13bb2ff6edbbb9f29aa786450a30a91/pandas-2.2.3-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl ; sys_platform == 'linux'`.



If you open a `pandas`'s wheel (unzip it), and go to the `dist-info` directory then open the `METADATA` file, you'll see which packages are required for which extra 😀, you should see things like `Requires-Dist: openpyxl>=3.1.0; extra == "excel"`. Keep in mind this distinction between `extra` in the context of the `METADATA` file and the extras in the context of something like `requests[security]`. It's also different from the context of the `pylock.toml` extension for which we'll talk about afterwards.



Obviously, the version clauses are mutually exclusive with the @ usage. The version is either tied to that direct reference (in the case of wheels and sdists for example) or in some cases you have to inspect the archive or tarball or whatever you're pointing to, or maybe even build it first.



The other thing to know about dependency specifiers, is that their grammar and them being a one-line requirement summary of your dependency means they're specify dependencies in a way that's independent from tools or how tools design their CLIs. Which makes sense, we're talking about specification. So you won't see some particular use cases talked about in the specification because that mainly falls in the realm of how tools interpret those use cases. For example editable installs. You can point to a dependency using the direct reference, but the specification says nothing about encoding that "editable" meaning in the direct reference. It's an installers' job (through an `-e` option for example). But, for name-based requirement, the index used to look up the files is still outside the specification. It's only the case if you use a direct reference. And one thing we didn't talk about, the specification recommends to include hashes for remote URLs.



And remember, deciding which wheel to install is the installer's job but at the end of the day if you ask from the installer a wheel for Windows while you're on Linux, the installer will give you an error. The installer will also try to find the best wheel after doing its dependency resolution (in case it does one and doesn't reuse one from the lock file) and that might fail if it doesn't fail a version that corresponds to what you want.



These two specifications should allow you to find the exact package you need and under different circumstances. But, they do not cover some use cases. The one that I care the most about is handling GPU, or more generally, accelerator, dependent packages, like `torch`. One quick way to see why these specifications (more like the dependency specifiers specification) fail at handling this, is not having a marker for the accelerator. But the issue is more complex than just adding a marker, that's why I'm dedicating a section for it.



Now, about the extension from PEP 751. Since PEP 751's lock files are able to hold my install scenarios (the multi-use lock files), there should be a way to describe what the user wants. You may have noticed that with the dependency specifiers above, but there is no way to encode something like "the user wants this and that dependency group as part of this install". The `extra` keyword that is only defined inside a containing layer like the `METADATA` file doesn't help with this scenario. Its goal is to record the information that a package is required only as part of some extra.  
  
So PEP 751 introduced four things as part of its extension, and at its core is the introduction of two new markers `extras` (noticed the plural form) and `dependency_groups`. These new variables can only be referenced in a lock file, and tools should raise an error if they encounter them outside of this context (maybe that will change in the future). The other two changes are that these variables hold sets instead of strings and that you can express comparison operators with the usual Python keywords, `in`, `not in`, `and` etc. So your set of `extras` in the lock file can be something like (example from the PEP itself): `extras = ["extra-1", "extra-2"]` and you can do something like `'extra-1' in extras or 'extra-2' in extras` to install a package when the user asked for either `'extra-1'` or `'extra-2'`.  
  
The default for `extras` when nothing is the empty set while for `dependency-groups` the installer *may* choose to default to the empty set or to the `default-groups`.  
  
Now that you know the essence of dependency specifiers and these new markers, you understand how the installer chooses which packages to install. For each `[[packages]]` entry, it evaluates the `packages.marker` key if it is present and uses the different markers to figure out whether it should install the package or not.
