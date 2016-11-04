package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"strconv"

	"github.com/unixpickle/mnist"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/skewcoder"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

const (
	DefaultStepSize = 0.001
	BatchSize       = 16
)

func main() {
	if len(os.Args) != 2 && len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "Usage:", os.Args[0], "<output_net> [step_size]")
		os.Exit(1)
	}

	stepSize := DefaultStepSize
	if len(os.Args) == 3 {
		var err error
		stepSize, err = strconv.ParseFloat(os.Args[2], 64)
		if err != nil {
			fmt.Fprintln(os.Stderr, "Invalid step size:", err)
			os.Exit(1)
		}
	}

	var net neuralnet.Network
	if netData, err := ioutil.ReadFile(os.Args[1]); err == nil {
		net, err = neuralnet.DeserializeNetwork(netData)
		if err != nil {
			fmt.Fprintln(os.Stderr, "Failed to decode network:", err)
			os.Exit(1)
		}
	} else {
		log.Println("Creating network...")
		net = createNetwork()
	}

	training := dataSetSamples(mnist.LoadTrainingDataSet())

	grad := &sgd.RMSProp{
		Gradienter: &neuralnet.BatchRGradienter{
			Learner:       net.BatchLearner(),
			CostFunc:      neuralnet.MeanSquaredCost{},
			MaxBatchSize:  BatchSize,
			MaxGoroutines: 1,
		},
		Resiliency: 0.9,
	}

	var iter int
	sgd.SGDMini(grad, training, stepSize, BatchSize, func(s sgd.SampleSet) bool {
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

func createNetwork() neuralnet.Network {
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
	return net
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
