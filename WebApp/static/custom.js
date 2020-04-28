// function to adjust the sudoku grid
function adjustSudokuBox() {
    $('.sudoku-box').each(function() {
            $(this).css('font-size', Math.max(0.3*parseFloat($(this).parent().css('width'))/9, 10));
            $(this).css('height', parseFloat($(this).parent().css('width'))/9)
        }
    )
}

function fillPuzzle(puzzle) {
    $('.sudoku').data('puzzle', JSON.stringify({'puzzle': puzzle}));
    if ($('.sudoku')[0].hasAttribute('data-solution')) {
        $('.sudoku').removeData('solution')
    }
    for (var i = 0; i < puzzle.length; i++) {
        for (var j = 0; j < puzzle[i].length; j++){
            $('#'.concat(i.toString(), '-', j.toString())).removeClass('error')
            if (puzzle[i][j] != 0){
                $('#'.concat(i.toString(), '-', j.toString())).html(puzzle[i][j].toString());
                $('#'.concat(i.toString(), '-', j.toString())).attr('contenteditable', 'false');
                $('#'.concat(i.toString(), '-', j.toString())).addClass('locked')
            }else{
                $('#'.concat(i.toString(), '-', j.toString())).html('');
                $('#'.concat(i.toString(), '-', j.toString())).attr('contenteditable', 'true')
                $('#'.concat(i.toString(), '-', j.toString())).removeClass('locked')
            }
        }
    }
}

function getPuzzle() {
    var puzzle = Array.from(Array(9), _ => Array(9).fill(0));
    $('.sudoku-box').each(function() {
        if ($(this).hasClass('locked')){
            var index = $(this).attr('id').split("");
            puzzle[index[0]][index[2]] = parseInt($(this).html())
        }
    });
    return puzzle
}


// adjust sudoku grid on load
$(document).ready(function() {
    adjustSudokuBox()
});

// adjsut webpage when screen size changes
$( window ).resize(function() {
    adjustSudokuBox();
    if ($( window ).width() < 768){
        $('.navbar').each(function() {
            $(this).attr('data-toggle', "collapse");
            $(this).attr('data-target', "#navbarHeader")
        })
    } else {
        $('.navbar').each(function() {
            $(this).removeAttr('data-toggle');
            $(this).removeAttr('data-target')
        })
    }
});

// Prevent users from entering more than one digit per field
$('.sudoku-box').on('keydown paste', function(event) {
    var allowed_keys = [49, 50, 51, 52, 53, 54, 56, 57]
    var always_allowed_keys = [8, 46]
    if(!(always_allowed_keys.includes(event.keyCode) ||($(this).text().length < 1 && allowed_keys.includes(event.keyCode)))) {
        event.preventDefault();
    } else {
        $(this).removeClass('error')
    }
});

//function to generate a newe Sudoku via a post request to the server
function generate(){
    $('#loading').find('h3').html('Generating Sudoku ...');
    $('#loading').css('display', 'flex');

    $.ajax({
        type: "POST",
        url: "/generateSudoku",
        success: generateSuccess,
        error: errorOccurred,
        dataType: "json"
    });

    function generateSuccess(data){
        fillPuzzle(data.puzzle);
        setTimeout(function () {$('#loading').css('display', 'none')}, 250)
    }
}

//function to solve a sudoku by sending it to the server, b solution type it can be determined if the complete solution is displayed or only the errors are highlighted.
function solve(solutionType){
    $('#loading').find('h3').html('Solving Sudoku ...');
    $('#loading').css('display', 'flex');

    var puzzle = getPuzzle()

    if (JSON.stringify({'puzzle': puzzle}) == $('.sudoku').data('puzzle') && $('.sudoku')[0].hasAttribute('data-solution')){
        solveSuccess(JSON.parse($('.sudoku').data('solution')))
    } else {
        $.ajax({
            type: "POST",
            url: "/solveSudoku",
            data: {'puzzle': JSON.stringify(puzzle)},
            success: solveSuccess,
            error: errorOccurred,
            dataType: "text"
        });
    }

    function solveSuccess(data) {
        $('.sudoku').data('puzzle', puzzle);
        $('.sudoku').data('solution', JSON.stringify('solution',JSON.parse(data).solution));

        var solution = JSON.parse(data).solution;

        for (var i = 0; i < solution.length; i++) {
            for (var j = 0; j < solution[i].length; j++){
                if (puzzle[i][j] != 0){
                    $('#'.concat(i.toString(), '-', j.toString())).html(puzzle[i][j].toString());
                    $('#'.concat(i.toString(), '-', j.toString())).attr('contenteditable', 'false');
                    $('#'.concat(i.toString(), '-', j.toString())).addClass('locked')
                }else{
                    if (solutionType == "check"){
                        if (solution[i][j] != $('#'.concat(i.toString(), '-', j.toString())).html() && $('#'.concat(i.toString(), '-', j.toString())).html() != ""){
                            $('#'.concat(i.toString(), '-', j.toString())).addClass('error')
                        }
                    } else {
                        $('#'.concat(i.toString(), '-', j.toString())).html(solution[i][j].toString());
                    }
                    $('#'.concat(i.toString(), '-', j.toString())).attr('contenteditable', 'true')
                    $('#'.concat(i.toString(), '-', j.toString())).removeClass('locked')
                }
            }
        }

        $('#loading').css('display', 'none')
    }
}



function errorOccurred(){
    alert("Sorry an error occured while fetching the data. Please reload the page and try again")
}