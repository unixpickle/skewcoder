package main

import (
	"errors"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

type FeatureDropper struct {
	Keep map[int]bool
}

func (f *FeatureDropper) Apply(in autofunc.Result) autofunc.Result {
	mask := make(linalg.Vector, len(in.Output()))
	for x := range mask {
		if f.Keep[x] {
			mask[x] = 1
		}
	}
	return autofunc.Mul(&autofunc.Variable{Vector: mask}, in)
}

func (f *FeatureDropper) ApplyR(rv autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	mask := make(linalg.Vector, len(in.Output()))
	for x := range mask {
		if f.Keep[x] {
			mask[x] = 1
		}
	}
	maskVar := autofunc.NewRVariable(&autofunc.Variable{Vector: mask}, rv)
	return autofunc.MulR(maskVar, in)
}

func (f *FeatureDropper) SerializerType() string {
	return ""
}

func (f *FeatureDropper) Serialize() ([]byte, error) {
	return nil, errors.New("not implemented")
}
