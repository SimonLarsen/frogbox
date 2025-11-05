---
hide:
- navigation
---

--8<-- "README.md:1:22"

## Creating a new project

Create a new project using the `frogbox` CLI tool:


=== "YAML"

    ```
    frogbox project new -d . -f yaml
    ```

=== "JSON"

    ```
    frogbox project new -d . -f json
    ```

Implement your model in `models`, your dataset in `datasets` and your pipeline configuration in `configs`.

Before training, configure your training system with accelerate:

```
accelerate config
```

Then launch the training pipeline:

```
accelerate launch -m frogbox.run -c configs/config.json
```

See example projects in the `examples` folder on [GitHub](https://github.com/SimonLarsen/frogbox/tree/main/examples).
