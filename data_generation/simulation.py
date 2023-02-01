import meep as mp
from meep.verbosity_mgr import Verbosity

_verbosity = Verbosity(mp.cvar, "meep", 1)


class FDTD:
    cname_to_meep = {
        "ex": mp.Ex,
        "ey": mp.Ey,
        "ez": mp.Ez,
        "hx": mp.Hx,
        "hy": mp.Hy,
        "hz": mp.Hz,
    }

    def __init__(
        self,
        extent=(5, 5, 5),
        design_extent=(5, 5, 5),
        resolution=20,
        lcen=1.0,
        n1=1.0,
        n2=1.5,
        src_components=["ex"],
        out_components=["ex", "ey", "ez"],
    ):
        self.resolution = resolution
        self.fcen = 1 / lcen
        self.dpml = lcen / 2
        self.extent = extent
        self.design_extent = design_extent
        self.cell = mp.Vector3(*(ext + 2 * self.dpml for ext in extent))
        self.n1 = mp.Medium(index=n1)
        self.n2 = mp.Medium(index=n2)
        self.src_components = src_components
        self.out_components = out_components

    @property
    def source_volume(self):
        if len(self.extent) == 2:
            return mp.Volume(
                center=mp.Vector3(-self.cell.x / 2 + self.dpml, 0),
                size=mp.Vector3(0, self.cell.y),
            )
        elif len(self.extent) == 3:
            return mp.Volume(
                center=mp.Vector3(0, 0, -self.cell.z / 2 + self.dpml),
                size=mp.Vector3(self.cell.x, self.cell.y, 0),
            )
        else:
            raise ValueError

    @property
    def design_region(self):
        return mp.Volume(center=mp.Vector3(), size=self.design_extent)

    def __call__(self, design):
        assert design.ndim == len(self.extent)

        design_variables = mp.MaterialGrid(
            mp.Vector3(*design.shape), self.n1, self.n2, grid_type="U_MEAN"
        )
        design_variables.update_weights(design.astype("f8"))

        geometry = [
            mp.Block(
                center=self.design_region.center,
                size=self.design_region.size,
                material=design_variables,
            )
        ]

        sources = [
            mp.Source(
                mp.ContinuousSource(
                    frequency=self.fcen,
                    is_integrated=True,
                    end_time=10,
                ),
                volume=self.source_volume,
                component=self.cname_to_meep[c],
            )
            for c in self.src_components
        ]

        sim = mp.Simulation(
            cell_size=self.cell,
            resolution=self.resolution,
            sources=sources,
            geometry=geometry,
            boundary_layers=[mp.PML(self.dpml)],
            k_point=mp.Vector3(),
        )

        dft = sim.add_dft_fields(
            [self.cname_to_meep[c] for c in self.out_components],
            [self.fcen],
            where=self.design_region,
        )

        sim.run(until_after_sources=mp.stop_when_dft_decayed())

        return {
            c: sim.get_dft_array(dft, self.cname_to_meep[c], 0)
            for c in self.out_components
        }
