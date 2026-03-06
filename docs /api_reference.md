# API Reference

## Main Classes

### `UA_PBR`
Main framework class.

```python
model = UA_PBR(config=None, **kwargs)
model.fit(train_loader, val_loader)
results = model.evaluate(test_loader)
