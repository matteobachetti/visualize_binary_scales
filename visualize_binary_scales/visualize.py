import os
import sys
import argparse
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from pint.models import get_model
from .orbit_utils import (
    get_m2_from_inclination_func,
    get_m2_from_inclination,
    a_from_a1,
    a2_from_a1,
    r_co,
    roche_lobe,
    get_inclination,
    get_eclipse_inclination,
)


def plot_system_scales(model, m1, inclination=None, eclipsing=None, m2max=100 * u.M_sun, r2=None):
    a1sini = model.A1.quantity
    porb = model.PB.quantity
    spin = (1 / model.F0.quantity).to(u.s)
    name = model.PSR.value

    if inclination is None:

        (incl_min, incl_max), m2_func = get_m2_from_inclination_func(m1, porb, a1sini, m2max=m2max)
        if not eclipsing:
            incl_max = get_eclipse_inclination(porb, a1sini, m1)
        print(incl_max.to(u.deg))
        m2min = m2_func(incl_max)

        nmax = max(np.log10(m2max / m2min).astype(int) * 2, 2)
        incl_vals = [
            get_inclination(porb, a1sini, m1, m2) for m2 in np.geomspace(m2min, m2max, nmax)
        ]

        for incl in incl_vals:
            plot_system_scales(model, m1, inclination=incl, eclipsing=eclipsing, m2max=m2max, r2=r2)
        return

    m2 = get_m2_from_inclination(
        inclination, porb, a1sini, m1, bracket=(0.001, m2max.to(u.M_sun).value)
    )
    print(m1, m2, porb, a1sini, spin)

    # Now, important radii
    a1 = (a1sini / np.sin(inclination)).to(u.km)
    orb_sep = a_from_a1(a1, m1, m2).to(u.km)
    a2 = a2_from_a1(a1, m1, m2).to(u.km)
    roche_lobe_radius = roche_lobe(m1, m2, a1).to(u.km)
    corotation_radius = r_co(spin, m1).to(u.km)
    ns_radius = 10 * u.km

    print(f"Orbital separation: {orb_sep}")
    print(f"A1: {a1}")
    print(f"A2: {a2}")
    print(f"Roche lobe radius: {roche_lobe_radius}")
    print(f"Corotation radius: {corotation_radius}")

    title = f"{name} -- M1: {m1.to(u.M_sun).value:.2f}, M2: {m2.to(u.M_sun).value:.2f}"
    angles = np.linspace(0, np.pi / 2, 100)
    x = np.sin(angles)
    y = np.cos(angles)
    plt.figure(title, figsize=(7, 3.5))
    gs = plt.GridSpec(1, 2)
    axscales = plt.subplot(gs[0])
    axobjects = plt.subplot(gs[1])

    def plot_radius(radius, label):
        axscales.plot(radius * x, radius * y, label=label)

    plot_radius(np.log10(orb_sep.value), "Orbit")
    plot_radius(np.log10(roche_lobe_radius.value), "Roche Lobe")
    plot_radius(np.log10(a1.value), "A1")
    # plot_radius(np.log10(a2.value), "Companion semi-major axis")
    plot_radius(np.log10(corotation_radius.value), "R_co")
    plot_radius(np.log10(ns_radius.value), r"$R_{\rm NS}$")

    axscales.set_xlabel(r"$\log_{10} \mathrm{length\,scale\,(km)}$")
    # axscales.set_ylabel(r"$\log_{10} d/\mathrm{km}$")
    plt.tight_layout()
    axscales.set_aspect("equal", adjustable="box")
    axscales.set_xlim([0, None])
    axscales.set_ylim([0, None])
    axscales.legend(loc="upper right")

    axobjects.plot(1e-6 * orb_sep.value * x, 1e-6 * orb_sep.value * y, label="Orbit")
    axobjects.plot(
        1e-6 * roche_lobe_radius.value * x, 1e-6 * roche_lobe_radius.value * y, label="Roche Lobe"
    )

    angle_45 = np.sin(np.pi / 4)
    axobjects.scatter(1e-6 * orb_sep.value * angle_45, 1e-6 * orb_sep.value * angle_45)

    axobjects.scatter(
        1e-6 * a2.value * angle_45,
        1e-6 * a2.value * angle_45,
        marker="x",
        label="CoM",
        color="k",
        s=50,
    )

    axobjects.plot([0, 1e-6 * orb_sep.value], [0, 1e-6 * orb_sep.value], "--", color="grey")

    axobjects.set_xlim([0, None])
    axobjects.set_ylim([0, None])
    axobjects.set_xlabel(r"$x (10^6 \mathrm{km})$")
    axobjects.set_ylabel(r"$y (10^6 \mathrm{km})$")
    axobjects.set_aspect("equal", adjustable="box")
    axobjects.legend(loc="upper right")

    plt.show()


def main(args=None):
    """Main function called by the `visualize_scales` command line script."""

    description = "Visualize the relevant scales of an accreting binary"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "-p",
        "--parfile",
        type=str,
        default=None,
        help="PINT-compatible model file",
    )
    parser.add_argument(
        "--eclipsing",
        type=bool,
        default=False,
        help="This is an eclipsing system",
    )
    parser.add_argument(
        "--m1",
        type=float,
        default=1.4,
        help="This is the accretor's mass in Solar masses",
    )
    parser.add_argument(
        "--r2",
        type=float,
        default=None,
        help="Radius of the donor in Solar radii",
    )
    parser.add_argument(
        "--m2max",
        type=float,
        default=100,
        help="This is the accretor's mass in Solar masses",
    )
    parser.add_argument(
        "--inclination",
        type=float,
        default=None,
        help=(
            "This is the orbital inclination (defaults to a range of values "
            "allowed for the system) in degrees. 90 is edge-on, 0 face-on"
        ),
    )

    parser.add_argument("--outfile", type=str, default=None, help="Output file name")

    args = parser.parse_args(args)
    args.m2max *= u.M_sun
    args.m1 *= u.M_sun
    if args.inclination is not None:
        args.inclination *= u.deg
    if args.r2 is not None:
        args.r2 *= u.R_sun
    model = get_model(args.parfile)
    plot_system_scales(
        model,
        args.m1,
        inclination=args.inclination,
        eclipsing=args.eclipsing,
        m2max=args.m2max,
        r2=args.r2,
    )


if __name__ == "__main__":
    main(args=sys.argv[1:])
