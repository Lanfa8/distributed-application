package schemas

type Action struct {
	Action string `json:"action"`
}

type SendFileAction struct {
	Action   string `json:"action"`
	FileName string `json:"fileName"`
	Filesize []byte `json:"filesize"`
}

type OtherAction struct {
	Action           string `json:"action"`
	OtherActionData1 string `json:"other_action_data_1"`
	OtherActionData2 string `json:"other_action_data_2"`
}
