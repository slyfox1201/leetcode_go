package main

func majorityElement1(nums []int) int {
	res := nums[0]
	cnt := 1
	for i := 1; i < len(nums); i++ {
		if cnt == 0 {
			res = nums[i]
		}
		if nums[i] == res {
			cnt++
		} else {
			cnt--
		}
	}
	return res
}

func majorityElement(nums []int) int {
	res, cnt := -1, 0
	for _, num := range nums {
		if cnt == 0 {
			res = num
		}
		if num == res {
			cnt++
		} else {
			cnt--
		}
	}
	return res
}
