def show_sim(data):
    """
    show an image
    """
    import matplotlib.pyplot as mplt
    from matplotlib.colors import AsinhNorm
    import lsst.geom as geom

    nim = 0
    for band in data:
        for se_obs in data[band]:
            nim += 1

    fig, axs = mplt.subplots(
        layout='tight',
        nrows=nim,
        ncols=2,
        figsize=(9, 4.0 * nim),
    )

    im_norm = AsinhNorm(
        linear_width=0.14,
        clip=False,
    )
    psf_norm = AsinhNorm(
        linear_width=0.01,
        clip=False,
    )

    pos = geom.Point2D(25.1, 31.5)
    im_index = 0
    for band in data:
        for epoch, se_obs in enumerate(data[band]):

            seimage = se_obs.image.array
            seimage = seimage / seimage.max()

            dmpsf = se_obs.getPsf()
            psf_image = dmpsf.computeImage(pos).array
            psf_image = psf_image / psf_image.max()

            plot_with_colorbar(
                fig=fig,
                ax=axs[im_index, 0],
                image=seimage,
                cmap='inferno',
                norm=im_norm,
                title=f'band: {band} epoch: {epoch}',
            )
            plot_with_colorbar(
                fig=fig,
                ax=axs[im_index, 1],
                image=psf_image,
                cmap='inferno',
                norm=psf_norm,
                title='PSF',
            )

            im_index += 1

    mplt.show()
    mplt.close(fig)


def plot_with_colorbar(fig, ax, image, title=None, **kw):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    pim = ax.imshow(image, **kw)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(pim, cax=cax)

    if title is not None:
        ax.set_title(title)
