def show_sim(data):
    """
    show an image
    """
    # from descwl_coadd.vis import show_images, show_2images
    # from espy import images
    import metadetect.lsst.vis

    # imlist = []
    for band in data:
        for iexp, exp in enumerate(data[band]):
            mess = f'band: {band} epoch: {iexp}'
            metadetect.lsst.vis.show_exp(exp, mess=mess, use_mpl=True)
            # sim = exp.image.array
            # sim = images.asinh_scale(image=sim/sim.max(), nonlinear=0.14)
            # imlist.append(sim)
            # imlist.append(exp.get_psf(25.1, 31.5).array)
            # imlist.append(exp.mask.array)

    # if len(imlist) == 2:
    #     show_2images(*imlist)
    # else:
    #     show_images(imlist)
