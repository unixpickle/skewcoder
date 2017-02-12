package main

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

type FeatureDropper struct {
	Keep  map[int]bool
	Means []float32
}

func (f *FeatureDropper) Apply(in anydiff.Res, n int) anydiff.Res {
	mask := make([]float32, in.Output().Len())
	means := make([]float32, in.Output().Len())
	for i := range mask {
		if f.Keep[i] {
			mask[i] = 1
		} else {
			means[i] = f.Means[i]
		}
	}
	c := in.Output().Creator()
	maskVec := repeat(c.MakeVectorData(mask), n)
	meansVec := repeat(c.MakeVectorData(means), n)
	return anydiff.Add(anydiff.Mul(anydiff.NewConst(maskVec), in),
		anydiff.NewConst(meansVec))
}

func repeat(v anyvec.Vector, n int) anyvec.Vector {
	res := v.Creator().MakeVector(v.Len() * n)
	anyvec.AddRepeated(res, v)
	return res
}
