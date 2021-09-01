package main

import "math"

type MinStack struct {
	stack1 []int
	stack2 []int
}

func Constructor() MinStack {
	return MinStack{
		stack1: []int{},
		stack2: []int{math.MaxInt32},
	}
}

func (this *MinStack) Push(x int) {
	this.stack1 = append(this.stack1, x)
	this.stack2 = append(this.stack2, this.IntMin(this.stack2[len(this.stack2)-1], x))
}

func (this *MinStack) Pop() {
	this.stack1 = this.stack1[:len(this.stack1)-1]
	this.stack2 = this.stack2[:len(this.stack2)-1]
}

func (this *MinStack) Top() int {
	return this.stack1[len(this.stack1)-1]
}

func (this *MinStack) Min() int {
	return this.stack2[len(this.stack2)-1]
}

func (this *MinStack) IntMin(x, y int) int {
	if x < y {
		return x
	}
	return y
}
