hfir.func @my_gelu(%x: tensor<?x768xf32>) -> tensor<?x768xf32> {
  %0 = hfir.mul(%x, %x) : tensor<?x768xf32>
  hfir.return %0 : tensor<?x768xf32>
}