use quartz_nbt::{snbt, NbtCompound};
use std::error::Error;
use voxel_cad::LittleBlueprint;

fn main() -> Result<(), Box<dyn Error>> {
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
    let root = snbt::parse(snbt)?;
    let little_blueprint = LittleBlueprint::try_from(root.clone())?;
    let root2: NbtCompound = LittleBlueprint::try_into(little_blueprint)?;
    assert_eq!(root, root2);
    println!("{:#?}", root2);
    Ok(())
}
