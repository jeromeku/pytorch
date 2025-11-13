# Torch Compiler Passes

This folder captures the sequence of compiler passes executed by `torch.compile`.
It summarizes Dynamo's front-end transformations and Inductor's
pre-, joint-, and post-grad passes with code references and runnable snippets.

* [Dynamo front-end](dynamo.md)
* [Inductor pre-grad](inductor_pre_grad.md)
* [Inductor joint graph](inductor_joint.md)
* [Inductor post-grad](inductor_post_grad.md)

Each document links to source files at commit
`3f3215be754e8c923c0f7adf5c0356efe074ff5c`.
