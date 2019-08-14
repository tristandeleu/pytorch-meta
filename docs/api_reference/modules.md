## MetaBatchNorm1d

```python
torchmeta.modules.MetaBatchNorm1d(num_features, eps=1e-05, momentum=0.1,
    affine=True, track_running_stats=True)
```

!!! note "Notes"
    See: `torch.nn.BatchNorm1d`

## MetaBatchNorm2d

```python
torchmeta.modules.MetaBatchNorm2d(num_features, eps=1e-05, momentum=0.1,
    affine=True, track_running_stats=True)
```

!!! note "Notes"
    See: `torch.nn.BatchNorm2d`

## MetaBatchNorm3d

```python
torchmeta.modules.MetaBatchNorm3d(num_features, eps=1e-05, momentum=0.1,
    affine=True, track_running_stats=True)
```

!!! note "Notes"
    See: `torch.nn.BatchNorm3d`

## MetaSequential

```python
torchmeta.modules.MetaSequential(*args)
```

!!! note "Notes"
    See: `torch.nn.Sequential`

## MetaConv1d

```python
torchmeta.modules.MetaConv1d(in_channels, out_channels, kernel_size, stride=1,
    padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
```

!!! note "Notes"
    See: `torch.nn.Conv1d`

## MetaConv2d

```python
torchmeta.modules.MetaConv2d(in_channels, out_channels, kernel_size, stride=1,
    padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
```

!!! note "Notes"
    See: `torch.nn.Conv2d`

## MetaConv3d

```python
torchmeta.modules.MetaConv3d(in_channels, out_channels, kernel_size, stride=1,
    padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
```

!!! note "Notes"
    See: `torch.nn.Conv3d`

## MetaLinear

```python
torchmeta.modules.MetaLinear(in_features, out_features, bias=True)
```

!!! note "Notes"
    See: `torch.nn.Linear`

## MetaBilinear

```python
torchmeta.modules.MetaBilinear(in1_features, in2_features, out_features,
    bias=True)
```

!!! note "Notes"
    See: `torch.nn.Bilinear`

## MetaModule

Base class for PyTorch meta-learning modules. These modules accept an additional argument `params` in their `forward` method.

```python
torchmeta.modules.MetaModule()
```

!!! note "Notes"
    Objects inherited from `MetaModule` are fully compatible with PyTorch modules from `torch.nn.Module`. The argument `params` is a dictionary of tensors, with full support of the computation graph (for differentiation).

## MetaLayerNorm

```python
torchmeta.modules.MetaLayerNorm(normalized_shape, eps=1e-05,
    elementwise_affine=True)
```

!!! note "Notes"
    See: `torch.nn.LayerNorm`