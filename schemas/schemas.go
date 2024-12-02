package schemas

type ActionType string

const (
	SEND_FILE_ACTION ActionType = "send_file"
	PROCESS_ACTION   ActionType = "process_action"
)

type Device string

const (
	GPU Device = "gpu"
	CPU Device = "cpu"
)

type Method string

const (
	OPENMP                 Method = "openmp"
	MPI                    Method = "mpi"
	GO_ROUTINES            Method = "go_routines"
	CUDA                   Method = "cuda"
	PYTHON_MULTIPROCESSING Method = "python_multiprocessing"
)

type Action struct {
	Action ActionType `json:"action"`
}

type SendFileAction struct {
	Action   ActionType `json:"action"`
	FileName string     `json:"fileName"`
	Filesize []byte     `json:"filesize"`
}

type ProcessAction struct {
	Action    ActionType `json:"action"`
	FileName  string     `json:"fileName"`
	Processes int        `json:"processes"`
	Method    Method     `json:"method"`
	Attempts  int        `json:"attempts"`
}
