# Voxel CAD Core

用于体素的精简CAD内核，将由多个部分构成：

## Part1 | LittleTiles SNBT 解析器

用于解析和序列化 Minecraft Mod LittleTiles SNBT 数据，将字符串化的 NBT格式转换为内部的 `LittleBlueprint` 结构，并能够将解析后的数据重新序列化回 NBT。

#### 使用示例

```rust
use quartz_nbt::{NbtCompound, snbt};
use voxel_cad::LittleBlueprint;

let root = snbt::parse(snbt)?;
let little_blueprint = LittleBlueprint::try_from(root.clone())?;
let root2: NbtCompound = LittleBlueprint::try_into(little_blueprint)?;
assert_eq!(root, root2);
println!("{:#?}", root2);
```

---

## Part2 | Voxel Engine

正在开发中

---

## 备注&附录:

暂无
