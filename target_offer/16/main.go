package main

func myPow(x float64, n int) float64 {
	if n > 0 {
		return helper(x, n)
	} else {
		return 1 / helper(x, -n)
	}
}

func helper(x float64, n int) float64 {
	var res float64 = 1
	for ; n != 0; n >>= 1 {
		if n&1 == 1 {
			res *= x
		}
		x *= x
	}
	return res
}
