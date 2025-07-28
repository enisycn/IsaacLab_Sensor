# Patch for adding gap terrain to SDS environment
# Add this to the end of velocity_env_cfg.py

# Import gap terrain generator
import isaaclab.terrains as terrain_gen

# Define gap test terrain  
GAP_TEST_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0, 
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "gap_terrain": terrain_gen.MeshGapTerrainCfg(
            proportion=0.7,
            gap_width_range=(0.4, 0.9), 
            platform_width=2.0,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3,
            noise_range=(0.05, 0.20),
            noise_step=0.02, 
            border_width=0.25
        ),
    },
)

# To enable gap terrain, change line ~53 in velocity_env_cfg.py from:
# terrain_generator=ROUGH_TERRAINS_CFG,
# to:
# terrain_generator=GAP_TEST_TERRAIN_CFG,
