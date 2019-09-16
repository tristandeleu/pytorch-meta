## Categorical

Target transform to return labels in `[0, num_classes)`.

```python
torchmeta.transforms.Categorical(num_classes=None)
```

**Parameters**

 - **num_classes**: *int, optional*
 Number of classes. If `None`, then the number of classes is inferred from the number of individual labels encountered.