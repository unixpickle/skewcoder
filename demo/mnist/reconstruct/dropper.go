package main

import (
	"errors"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

type FeatureDropper struct {
	Keep  map[int]bool
	Means linalg.Vector
}

func (f *FeatureDropper) Apply(in autofunc.Result) autofunc.Result {
	mask := make(linalg.Vector, len(in.Output()))
	means := make(linalg.Vector, len(in.Output()))
	for x := range mask {
		if f.Keep[x] {
			mask[x] = 1
		} else {
			means[x] = f.Means[x]
		}
	}
	return autofunc.Add(autofunc.Mul(&autofunc.Variable{Vector: mask}, in),
		&autofunc.Variable{Vector: means})
}

func (f *FeatureDropper) ApplyR(rv autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	mask := make(linalg.Vector, len(in.Output()))
	means := make(linalg.Vector, len(in.Output()))
	for x := range mask {
		if f.Keep[x] {
			mask[x] = 1
		} else {
			means[x] = f.Means[x]
		}
	}
	maskVar := autofunc.NewRVariable(&autofunc.Variable{Vector: mask}, rv)
	meanVar := autofunc.NewRVariable(&autofunc.Variable{Vector: means}, rv)
	return autofunc.AddR(autofunc.MulR(maskVar, in), meanVar)
}

func (f *FeatureDropper) SerializerType() string {
	return ""
}

func (f *FeatureDropper) Serialize() ([]byte, error) {
	return nil, errors.New("not implemented")
}
