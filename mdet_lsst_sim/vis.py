def show_sim(data):
    """
    show an image
    """
    from descwl_coadd.vis import show_images, show_2images
    from espy import images

    imlist = []
    for band in data:
        for se_obs in data[band]:
            sim = se_obs.image.array
            sim = images.asinh_scale(image=sim/sim.max(), nonlinear=0.14)
            imlist.append(sim)
            imlist.append(se_obs.get_psf(25.1, 31.5).array)

    if len(imlist) == 2:
        show_2images(*imlist)
    else:
        show_images(imlist)
