package main

import (
	"image/png"
	"log"
	"os"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/mnist"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/skewcoder"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

func main() {
	net := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  28 * 28,
			OutputCount: 50,
		},
		&neuralnet.HyperbolicTangent{},
		&skewcoder.Reconstructor{
			Block: rnn.NewLSTM(1, 300),
		},
		&neuralnet.DenseLayer{
			InputCount:  300,
			OutputCount: 28 * 28,
		},
		neuralnet.Sigmoid{},
	}
	net.Randomize()
	training := dataSetSamples(mnist.LoadTrainingDataSet())

	grad := &sgd.RMSProp{
		Gradienter: &neuralnet.BatchRGradienter{
			Learner:  net.BatchLearner(),
			CostFunc: neuralnet.MeanSquaredCost{},
		},
		Resiliency: 0.9,
	}

	var iter int
	sgd.SGDMini(grad, training, 0.001, 16, func(s sgd.SampleSet) bool {
		cost := neuralnet.TotalCost(neuralnet.MeanSquaredCost{}, net, s)
		log.Printf("iter %d: cost=%f", iter, cost)
		iter++
		return true
	})

	img := mnist.ReconstructionGrid(func(img []float64) []float64 {
		v := &autofunc.Variable{Vector: img}
		return net.Apply(v).Output()
	}, mnist.LoadTestingDataSet(), 5, 5)

	f, _ := os.Create("output.png")
	defer f.Close()
	png.Encode(f, img)
}

func dataSetSamples(d mnist.DataSet) sgd.SampleSet {
	inputVecs := vecVec(d.IntensityVectors())
	return neuralnet.VectorSampleSet(inputVecs, inputVecs)
}

func vecVec(f [][]float64) []linalg.Vector {
	res := make([]linalg.Vector, len(f))
	for i, x := range f {
		res[i] = x
	}
	return res
}
