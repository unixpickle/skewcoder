package skewcoder

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/rnn"
)

func TestReconstructor(t *testing.T) {
	f := &Reconstructor{
		Block: rnn.NewLSTM(1, 3),
	}
	inVar := &autofunc.Variable{Vector: []float64{1, -1, 2, -0.5, 3}}
	allVars := append([]*autofunc.Variable{inVar}, f.Parameters()...)
	rv := autofunc.RVector{}
	for _, v := range allVars {
		r := make(linalg.Vector, len(v.Vector))
		for i := range r {
			r[i] = rand.NormFloat64()
		}
		rv[v] = r
	}
	checker := functest.RFuncChecker{
		F:     f,
		Vars:  allVars,
		Input: inVar,
		RV:    rv,
	}
	checker.FullCheck(t)
}
