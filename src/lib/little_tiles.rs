use bitflags::bitflags;
use enum_map::{Enum, EnumMap, enum_map};
use quartz_nbt::{NbtCompound, NbtList, NbtTag};
use std::{collections::HashMap, hash::Hash};

/// Error type for parsing and serialization
#[derive(Debug)]
pub enum ParseError {
    InvalidFormat,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::InvalidFormat => write!(f, "Invalid SNBT format"),
        }
    }
}

impl std::error::Error for ParseError {}

/// 坐标
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LittlePos {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct LittleColor {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl TryFrom<i32> for LittleColor {
    type Error = ParseError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        let c = value as u32;
        Ok(LittleColor {
            r: (c >> 24) as u8,
            g: (c >> 16) as u8,
            b: (c >> 8) as u8,
            a: c as u8,
        })
    }
}

impl TryInto<i32> for LittleColor {
    type Error = ParseError;

    fn try_into(self) -> Result<i32, Self::Error> {
        Ok(((self.r as i32) << 24)
            | ((self.g as i32) << 16)
            | ((self.b as i32) << 8)
            | (self.a as i32))
    }
}

/// 朝向
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Facing {
    Down,
    Up,
    North,
    South,
    West,
    East,
}

/// 立方体的 8 个角
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Enum)]
pub enum BoxCorner {
    EUN, // East, Up, North
    EUS, // East, Up, South
    EDN, // East, Down, North
    EDS, // East, Down, South
    WUN, // West, Up, North
    WUS, // West, Up, South
    WDN, // West, Down, North
    WDS, // West, Down, South
}

const CORNER_ORDER: [BoxCorner; 8] = [
    BoxCorner::EUN,
    BoxCorner::EUS,
    BoxCorner::EDN,
    BoxCorner::EDS,
    BoxCorner::WUN,
    BoxCorner::WUS,
    BoxCorner::WDN,
    BoxCorner::WDS,
];

/// 坐标轴枚举：X/Y/Z
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Enum)]
pub enum Axis {
    X,
    Y,
    Z,
}

bitflags! {
    /// 反转坐标轴
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Flipped: u8 {
        const EAST  = 0b0000_01;
        const WEST  = 0b0000_10;
        const SOUTH = 0b0001_00;
        const NORTH = 0b0010_00;
        const UP    = 0b0100_00;
        const DOWN  = 0b1000_00;
    }
}

/// 角落偏移量 8 * 3 = 24
type CornerOffsets = EnumMap<BoxCorner, EnumMap<Axis, i16>>;

/// Main tile enum
#[derive(Debug, Clone, PartialEq)]
pub enum LittleTile {
    Box {
        min_pos: LittlePos,
        max_pos: LittlePos,
    },
    TransformableBox {
        min_pos: LittlePos,
        max_pos: LittlePos,
        flips: Flipped,
        corner: CornerOffsets,
    },
}

fn get_int_field(nbt: &NbtCompound, field: &str) -> Result<i32, ParseError> {
    match nbt.inner().get(field) {
        Some(NbtTag::Int(value)) => Ok(*value),
        _ => Err(ParseError::InvalidFormat),
    }
}

fn get_int_array(nbt: &NbtCompound, field: &str) -> Result<Vec<i32>, ParseError> {
    match nbt.inner().get(field) {
        Some(NbtTag::IntArray(value)) => Ok(value.clone()),
        _ => Err(ParseError::InvalidFormat),
    }
}

// 解析变换数据
fn decode_transformable_data(data: &[i32]) -> Result<(Flipped, CornerOffsets), ParseError> {
    if data.is_empty() {
        return Err(ParseError::InvalidFormat);
    }
    // 计算Flipped位
    let flags_bits = data[0] as u32;
    let flips = Flipped::from_bits_truncate(((flags_bits >> 24) & 0x3F) as u8);

    // 计算偏移量
    let mut corner_offsets: CornerOffsets = enum_map! { _ => enum_map! { _ => 0 } };

    let mut vals = Vec::new();
    for &x in &data[1..] {
        let u = x as u32;
        vals.push((u >> 16) as i16);
        vals.push((u & 0xFFFF) as i16);
    }
    let mut vi = 0;
    for (ax_i, &axis) in [Axis::X, Axis::Y, Axis::Z].iter().enumerate() {
        for (corner_i, &corner) in CORNER_ORDER.iter().enumerate() {
            let bit = 3 * corner_i + ax_i;
            if ((flags_bits) >> bit) & 0x1 == 1 {
                if vi >= vals.len() {
                    return Err(ParseError::InvalidFormat);
                }
                corner_offsets[corner][axis] = vals[vi];
                vi += 1;
            }
        }
    }
    Ok((flips, corner_offsets))
}

// 编码变换数据
fn encode_transformable_data(
    flips: Flipped,
    corner_offsets: &CornerOffsets,
) -> Result<Vec<i32>, ParseError> {
    let mut flags_bits: u32 = 0;
    let mut data: Vec<i16> = Vec::new();

    // 与 decode 完全相同的遍历顺序
    for (corner_i, &corner) in CORNER_ORDER.iter().enumerate() {
        for (ax_i, &axis) in [Axis::X, Axis::Y, Axis::Z].iter().enumerate() {
            let offset = corner_offsets[corner][axis];
            if offset != 0 {
                flags_bits |= 1 << (3 * corner_i + ax_i);
                data.push(offset);
            }
        }
    }

    // 计算存储单元数量（每 2 个 i16 装进一个 i32）
    let total_words = (1 + data.len()) >> 1;
    let mut result = vec![0; 1 + total_words];

    // 组装首字
    let mut word0: u32 = 0x8000_0000; // magic bit
    word0 |= (flips.bits() as u32) << 24; // 6 个翻转位
    word0 |= flags_bits; // 偏移标志
    result[0] = word0 as i32;

    // 打包偏移量
    for i in 0..total_words {
        let hi = (data[i * 2] as u16 as i32) << 16;
        let lo = if 2 * i + 1 < data.len() {
            data[i * 2 + 1] as u16 as i32
        } else {
            0
        };
        result[i + 1] = hi | lo;
    }
    Ok(result)
}

impl TryFrom<Vec<i32>> for LittleTile {
    type Error = ParseError;

    fn try_from(arr: Vec<i32>) -> Result<Self, Self::Error> {
        let arr = arr.as_slice();
        // helper: 拆出 bbox 并返回剩余切片
        fn split_bbox(s: &[i32]) -> Option<(LittlePos, LittlePos, &[i32])> {
            if s.len() < 6 {
                return None;
            }
            let (head, rest) = s.split_at(6);
            let [min_x, min_y, min_z, max_x, max_y, max_z] = <[i32; 6]>::try_from(head).ok()?;
            let min_pos = LittlePos {
                x: min_x,
                y: min_y,
                z: min_z,
            };
            let max_pos = LittlePos {
                x: max_x,
                y: max_y,
                z: max_z,
            };
            Some((min_pos, max_pos, rest))
        }

        match arr.len() {
            6 => {
                let (min_pos, max_pos, _) = split_bbox(arr).ok_or(ParseError::InvalidFormat)?;
                Ok(LittleTile::Box { min_pos, max_pos })
            }
            n if n >= 7 => {
                let (min_pos, max_pos, rest) = split_bbox(arr).ok_or(ParseError::InvalidFormat)?;
                let (flips, corner) = decode_transformable_data(rest)?;
                Ok(LittleTile::TransformableBox {
                    min_pos,
                    max_pos,
                    flips,
                    corner,
                })
            }
            _ => Err(ParseError::InvalidFormat),
        }
    }
}

impl TryInto<Vec<i32>> for LittleTile {
    type Error = ParseError;

    fn try_into(self) -> Result<Vec<i32>, Self::Error> {
        match self {
            LittleTile::Box { min_pos, max_pos } => {
                let arr = [
                    min_pos.x, min_pos.y, min_pos.z, max_pos.x, max_pos.y, max_pos.z,
                ];
                Ok(arr.to_vec())
            }
            LittleTile::TransformableBox {
                min_pos,
                max_pos,
                flips,
                corner,
            } => {
                let mut arr = vec![
                    min_pos.x, min_pos.y, min_pos.z, max_pos.x, max_pos.y, max_pos.z,
                ];
                let corner_offsets = encode_transformable_data(flips, &corner)?;
                arr.extend(corner_offsets);
                Ok(arr)
            }
        }
    }
}

type ColorTiles = HashMap<LittleColor, Vec<LittleTile>>;
type Material = String;

type MaterialTiles = HashMap<Material, ColorTiles>;

#[derive(Debug, Clone, PartialEq)]
pub struct LittleGroup {
    pub grid: u16,
    pub children: Vec<LittleGroup>,
    pub tiles: MaterialTiles,
    pub structure: Option<NbtCompound>,
    pub extension: Option<NbtCompound>,
}

impl TryFrom<NbtCompound> for LittleGroup {
    type Error = ParseError;

    fn try_from(nbt: NbtCompound) -> Result<Self, Self::Error> {
        let mut map: HashMap<String, NbtTag> = nbt.into_inner();

        // 解析精度
        let Some(NbtTag::Int(grid)) = map.remove("grid") else {
            return Err(ParseError::InvalidFormat);
        };
        let grid = grid as u16;

        // 解析子组
        let mut children = Vec::new();
        let clist = match map.remove("c") {
            Some(NbtTag::List(list)) => list.into_inner(),
            None => Vec::new(),
            _ => return Err(ParseError::InvalidFormat),
        };
        for item in clist {
            let NbtTag::Compound(child) = item else {
                return Err(ParseError::InvalidFormat);
            };
            children.push(LittleGroup::try_from(child)?);
        }

        // 解析结构体
        let structure = match map.remove("s") {
            Some(NbtTag::Compound(c)) => Some(c),
            None => None,
            _ => return Err(ParseError::InvalidFormat),
        };

        // 解析扩展
        let extension = match map.remove("e") {
            Some(NbtTag::Compound(c)) => Some(c),
            None => None,
            _ => return Err(ParseError::InvalidFormat),
        };

        // 解析小方块
        let mut tiles: MaterialTiles = MaterialTiles::new();
        let Some(NbtTag::Compound(mt)) = map.remove("t") else {
            return Err(ParseError::InvalidFormat);
        };
        for (mat, tag) in mt.into_inner() {
            let NbtTag::List(flat_list) = tag else {
                return Err(ParseError::InvalidFormat);
            };
            let mut color_tiles: ColorTiles = HashMap::new();
            let mut cur_color = LittleColor::default();
            for tag in flat_list.into_inner() {
                match tag {
                    NbtTag::IntArray(ar) if ar.len() == 1 => {
                        cur_color = LittleColor::try_from(ar[0])?;
                    }
                    NbtTag::IntArray(ar) => {
                        let tile = LittleTile::try_from(ar)?;
                        color_tiles.entry(cur_color).or_default().push(tile);
                    }
                    _ => {
                        return Err(ParseError::InvalidFormat);
                    }
                }
            }
            tiles.insert(mat.clone(), color_tiles);
        }

        Ok(LittleGroup {
            grid,
            children,
            tiles,
            structure,
            extension,
        })
    }
}

impl TryInto<NbtCompound> for LittleGroup {
    type Error = ParseError;

    fn try_into(self) -> Result<NbtCompound, Self::Error> {
        let mut nbt = NbtCompound::new();

        // grid
        nbt.insert("grid", self.grid as i32);

        // children list
        let mut clist = Vec::new();
        for child in self.children {
            let child_nbt = LittleGroup::try_into(child)?;
            clist.push(NbtTag::Compound(child_nbt));
        }
        nbt.insert("c", NbtTag::List(NbtList::from(clist)));

        // optional structure
        if let Some(ref struct_c) = self.structure {
            nbt.insert("s", NbtTag::Compound(struct_c.clone()));
        }

        // optional extension
        if let Some(ref ext_c) = self.extension {
            nbt.insert("e", NbtTag::Compound(ext_c.clone()));
        }

        // tiles by material
        let mut mt = NbtCompound::new();
        for (mat, color_tiles) in &self.tiles {
            let mut flat = Vec::new();
            for (color, tiles) in color_tiles {
                // color marker
                let c_val: i32 = (*color).try_into()?;
                flat.push(NbtTag::IntArray(vec![c_val]));

                // each tile array
                for tile in tiles {
                    let arr: Vec<i32> = tile.clone().try_into()?;
                    flat.push(NbtTag::IntArray(arr));
                }
            }
            mt.insert(mat.clone(), NbtTag::List(NbtList::from(flat)));
        }
        nbt.insert("t", NbtTag::Compound(mt));

        Ok(nbt)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LittleBlueprint {
    pub boxes_cnt: u32,
    pub tiles_cnt: u32,
    pub min_pos: LittlePos,
    pub max_pos: LittlePos,
    pub top_group: LittleGroup,
}

impl TryFrom<NbtCompound> for LittleBlueprint {
    type Error = ParseError;

    fn try_from(root: NbtCompound) -> Result<Self, Self::Error> {
        let boxes_cnt = get_int_field(&root, "boxes")? as u32;
        let tiles_cnt = get_int_field(&root, "tiles")? as u32;
        let min_arr = get_int_array(&root, "min")?;
        let size_arr = get_int_array(&root, "size")?;
        if min_arr.len() != 3 || size_arr.len() != 3 {
            return Err(ParseError::InvalidFormat);
        }
        let min_pos = LittlePos {
            x: min_arr[0],
            y: min_arr[1],
            z: min_arr[2],
        };
        let max_pos = LittlePos {
            x: min_pos.x + size_arr[0],
            y: min_pos.y + size_arr[1],
            z: min_pos.z + size_arr[2],
        };
        // root group shares same shape as any other group
        let top_group = LittleGroup::try_from(root)?;
        Ok(LittleBlueprint {
            boxes_cnt,
            tiles_cnt,
            min_pos,
            max_pos,
            top_group,
        })
    }
}

impl TryInto<NbtCompound> for LittleBlueprint {
    type Error = ParseError;

    fn try_into(self) -> Result<NbtCompound, Self::Error> {
        // Helper: serialize a LittleGroup into an NbtCompound

        // Build the root compound from the top_group
        let mut root: NbtCompound = LittleGroup::try_into(self.top_group)?;

        // Blueprint metadata
        root.insert("boxes", NbtTag::Int(self.boxes_cnt as i32));
        root.insert("tiles", NbtTag::Int(self.tiles_cnt as i32));
        root.insert(
            "min",
            NbtTag::IntArray(vec![self.min_pos.x, self.min_pos.y, self.min_pos.z]),
        );
        let size_vec = vec![
            self.max_pos.x - self.min_pos.x,
            self.max_pos.y - self.min_pos.y,
            self.max_pos.z - self.min_pos.z,
        ];
        root.insert("size", NbtTag::IntArray(size_vec));

        Ok(root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quartz_nbt::snbt;

    #[test]
    fn test_encode_transformable_data() {
        let ar = [-2147475454, -65538];
        let (flips, corner_offsets) = decode_transformable_data(&ar).expect("Failed to decode");
        let ar_cur = encode_transformable_data(flips, &corner_offsets).expect("Failed to encode");
        assert_eq!(ar, ar_cur.as_slice());
    }

    #[test]
    fn test_blueprint() {
        let snbt = r#"
    {
        min: [I; 0, 0, 3],
        c: [
            {
                s: {
                    id: "fixed"
                },
                c: [],
                t: {
                    "minecraft:stone": [
                        [I; -1],
                        [I; 3, 0, 3, 4, 1, 4],
                        [I; 3, 0, 4, 4, 1, 5],
                        [I; 4, 0, 3, 5, 1, 4],
                        [I; 4, 0, 4, 5, 1, 5]
                    ]
                },
                grid: 4
            },
            {
                t: {
                    "minecraft:red_wool": [
                        [I; -1],
                        [I; 2, 0, 6, 3, 1, 7]
                    ]
                },
                c: [
                    {
                        c: [
                            {
                                grid: 4,
                                s: {
                                    id: "fixed"
                                },
                                c: [],
                                t: {
                                    "minecraft:lime_wool": [
                                        [I; -1],
                                        [I; 0, 0, 4, 1, 1, 5]
                                    ]
                                }
                            }
                        ],
                        t: {
                            "minecraft:purple_wool": [
                                [I; -1],
                                [I; 1, 0, 5, 2, 1, 6]
                            ]
                        },
                        grid: 4,
                        s: {
                            id: "fixed"
                        }
                    }
                ],
                grid: 4,
                s: {
                    id: "fixed"
                }
            }
        ],
        boxes: 8,
        tiles: 5,
        grid: 4,
        t: {
            "minecraft:white_wool": [
                [I; -1],
                [I; 3, 0, 7, 4, 1, 8]
            ]
        },
        size: [I; 5, 1, 5]
    }
        "#;
        let root = snbt::parse(snbt).expect("Failed to parse SNBT");
        let little_blueprint = LittleBlueprint::try_from(root.clone())
            .expect("Failed to convert SNBT to LittleBlueprint");
        let root2: NbtCompound = LittleBlueprint::try_into(little_blueprint)
            .expect("Failed to convert LittleBlueprint to SNBT");
        assert_eq!(root, root2);
    }
}
