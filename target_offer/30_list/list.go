package main

import (
	"container/list"
	"math"
)

type MinStack struct {
	stack1 *list.List
	stack2 *list.List
}

func Constructor() MinStack {
	minStack := MinStack{
		stack1: list.New(),
		stack2: list.New(),
	}
	minStack.stack2.PushBack(math.MaxInt32)
	return minStack
}

func (this *MinStack) Push(x int) {
	this.stack1.PushBack(x)
	this.stack2.PushBack(this.IntMin(this.stack2.Back().Value.(int), x))
}

func (this *MinStack) Pop() {
	this.stack1.Remove(this.stack1.Back())
	this.stack2.Remove(this.stack2.Back())
}

func (this *MinStack) Top() int {
	return this.stack1.Back().Value.(int)
}

func (this *MinStack) Min() int {
	return this.stack2.Back().Value.(int)
}

func (this *MinStack) IntMin(x, y int) int {
	if x < y {
		return x
	}
	return y
}
