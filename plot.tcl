proc drawGuides {lx ly hx hy layer} {
    add_gui_shape -layer $layer -rect [dbu2uu "$lx $ly $hx $hy"]
}