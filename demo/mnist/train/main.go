package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"

	"github.com/unixpickle/mnist"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/skewcoder"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage:", os.Args[0], "<output_net>")
		os.Exit(1)
	}

	net := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  28 * 28,
			OutputCount: 200,
		},
		&neuralnet.HyperbolicTangent{},
		&neuralnet.DenseLayer{
			InputCount:  200,
			OutputCount: 30,
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

	data, err := net.Serialize()
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to serialize output:", err)
		os.Exit(1)
	}
	if err := ioutil.WriteFile(os.Args[1], data, 0755); err != nil {
		fmt.Fprintln(os.Stderr, "Failed to write output:", err)
		os.Exit(1)
	}
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
