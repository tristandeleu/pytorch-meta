## MetaMultiheadAttention

```python
torchmeta.modules.MetaMultiheadAttention(*args, **kwargs)
```

!!! note "Notes"
    See: `torch.nn.MultiheadAttention`

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
torchmeta.modules.MetaSequential(*args:Any)
```

!!! note "Notes"
    See: `torch.nn.Sequential`

## MetaConv1d

```python
torchmeta.modules.MetaConv1d(in_channels:int, out_channels:int,
    kernel_size:Union[int, Tuple[int]], stride:Union[int, Tuple[int]]=1,
    padding:Union[int, Tuple[int]]=0, dilation:Union[int, Tuple[int]]=1,
    groups:int=1, bias:bool=True, padding_mode:str='zeros')
```

!!! note "Notes"
    See: `torch.nn.Conv1d`

## MetaConv2d

```python
torchmeta.modules.MetaConv2d(in_channels:int, out_channels:int,
    kernel_size:Union[int, Tuple[int, int]], stride:Union[int, Tuple[int,
    int]]=1, padding:Union[int, Tuple[int, int]]=0, dilation:Union[int,
    Tuple[int, int]]=1, groups:int=1, bias:bool=True,
    padding_mode:str='zeros')
```

!!! note "Notes"
    See: `torch.nn.Conv2d`

## MetaConv3d

```python
torchmeta.modules.MetaConv3d(in_channels:int, out_channels:int,
    kernel_size:Union[int, Tuple[int, int, int]], stride:Union[int, Tuple[int,
    int, int]]=1, padding:Union[int, Tuple[int, int, int]]=0,
    dilation:Union[int, Tuple[int, int, int]]=1, groups:int=1, bias:bool=True,
    padding_mode:str='zeros')
```

!!! note "Notes"
    See: `torch.nn.Conv3d`

## MetaLinear

```python
torchmeta.modules.MetaLinear(in_features:int, out_features:int,
    bias:bool=True) -> None
```

!!! note "Notes"
    See: `torch.nn.Linear`

## MetaBilinear

```python
torchmeta.modules.MetaBilinear(in1_features:int, in2_features:int,
    out_features:int, bias:bool=True) -> None
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
torchmeta.modules.MetaLayerNorm(normalized_shape:Union[int, List[int],
    torch.Size], eps:float=1e-05, elementwise_affine:bool=True) -> None
```

!!! note "Notes"
    See: `torch.nn.LayerNorm`

## DataParallel

```python
torchmeta.modules.DataParallel(module, device_ids=None, output_device=None,
    dim=0)
```

!!! note "Notes"
    See: `torch.nn.Parallel`

## MetaEmbedding

```python
torchmeta.modules.MetaEmbedding(num_embeddings:int, embedding_dim:int,
    padding_idx:Union[int, NoneType]=None, max_norm:Union[float,
    NoneType]=None, norm_type:float=2.0, scale_grad_by_freq:bool=False,
    sparse:bool=False, _weight:Union[torch.Tensor, NoneType]=None) -> None
```

!!! note "Notes"
    See: `torch.nn.Embedding`

## MetaEmbeddingBag

```python
torchmeta.modules.MetaEmbeddingBag(num_embeddings:int, embedding_dim:int,
    max_norm:Union[float, NoneType]=None, norm_type:float=2.0,
    scale_grad_by_freq:bool=False, mode:str='mean', sparse:bool=False,
    _weight:Union[torch.Tensor, NoneType]=None,
    include_last_offset:bool=False) -> None
```

!!! note "Notes"
    See: `torch.nn.EmbeddingBag`