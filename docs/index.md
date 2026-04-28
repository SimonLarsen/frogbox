---
hide:
- navigation
---

--8<-- "README.md:1:22"

## Creating a new project

Create a configuration file using the `frogbox` CLI tool:


=== "YAML"

    ```
    frogbox project new -f yaml -o config.yml
    ```

=== "JSON"

    ```
    frogbox project new -f json -o config.json
    ```

Implement your model as a `torch.nn.Module` and your dataset as a `torch.utils.data.Dataset` and configure the training pipeline by editing the newly created config file.

Before training, configure your training system with accelerate:

```
accelerate config
```

Then launch the training pipeline:

```
accelerate launch -m frogbox.run -c config.yml
```

See example projects in the `examples` folder on [GitHub](https://github.com/SimonLarsen/frogbox/tree/main/examples).
