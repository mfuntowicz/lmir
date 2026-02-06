lmir.func @my_gelu(%x: tensor<?x768xf32>) -> tensor<?x768xf32> {
  %0 = lmir.mul(%x, %x) : tensor<?x768xf32>
  lmir.return %0 : tensor<?x768xf32>
}
