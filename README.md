# EM Loop

Real-time 3D CUDA/OpenGL visualization of the exact vacuum Hopfion field from the Bateman/Riemann-Silberstein construction, rendered as traced electric and magnetic field lines inside an energy-density halo with bloom and a free-fly camera.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```

## Run

```bash
./build/em_loop
./build/em_loop --quality high
./build/em_loop --preset trefoil
./build/em_loop --field torus --p 2 --q 3
```

Available quality presets: `low`, `medium`, `high`, `ultra`
Available field modes: `hopfion`, `torus`
Available named presets: `hopfion`, `trefoil`, `cinquefoil`, `linked-rings`

## Controls

- `W`, `A`, `S`, `D`: move on the horizontal plane
- `Q`, `E`: move down / up
- Mouse: look around
- `Shift`: move faster
- `Tab`: toggle mouse capture
- `R`: reset camera and simulation time
- `1`, `2`, `3`: switch between `hopfion`, `trefoil`, and `cinquefoil`
- `F`: cycle post-process filters, including `OFF` and `NO GLOW`
- `Escape`: quit

## Notes

- The field evaluator is based on the exact Hopfion formula from Kedia et al., "Tying knots in light fields" (2013), using the Riemann-Silberstein vector `F = E + iB`.
- `--field torus --p P --q Q` switches to the exact Bateman torus-knot family `F = ∇α^P × ∇β^Q` from the same paper. Coprime `(P, Q)` values give torus knots; non-coprime pairs give linked rings.
- `--preset` is a convenience layer on top of the exact fields: `trefoil` = `(2,3)`, `cinquefoil` = `(2,5)`, and `linked-rings` = `(2,2)`.
- CUDA traces samples along the instantaneous electric and magnetic field lines each frame; the renderer visualizes those exact field directions, not a handcrafted flow.
- The top-left HUD and window title show the current quality, preset, field family, and FPS without requiring any extra text-rendering dependency.
- This is still a visualization layer, not a numerical Maxwell solver: colors, halo density mapping, time scaling, and streamline seeding are chosen for legibility and performance.
- `medium` is now the default preset and is tuned to be much lighter than the original first pass. Use `high` or `ultra` only if your frame time has headroom.
